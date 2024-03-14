import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 2001  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step)

    return image_tensor_cd


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    bear_tokens = tokenizer.tokenize("bear")
    man_tokens = tokenizer.tokenize("man")
    human_tokens = tokenizer.tokenize("human")
    person_tokens = tokenizer.tokenize("person")

    # Convert tokens to IDs
    bear_tokens_ids = tokenizer.convert_tokens_to_ids(bear_tokens)
    man_tokens_ids = tokenizer.convert_tokens_to_ids(man_tokens)
    human_tokens_ids = tokenizer.convert_tokens_to_ids(human_tokens)
    person_tokens_ids = tokenizer.convert_tokens_to_ids(person_tokens)
    noise_steps = list(range(0, 2000 + 1, 100))
    step_to_probs_dicts = []
    for noise_step in noise_steps:
        image_file = "debug/images/bear_horse.jpg"
        qs = "What sits on the back of the horse? Answer with a single word."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # prompt += " There is a brown horse, and on its back sits a "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = add_diffusion_noise(image_tensor, noise_step)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        input_token_len = input_ids.shape[1]
        scores_per_token= [score.softmax(dim=1).flatten() for score in outputs.scores]
        top_k_values, top_k_indices = scores_per_token[0].topk(20, dim=-1)

        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())
        topk_id_value_map = [{t.replace("▁", ""):v.item()} for t,v in zip(top_k_tokens, top_k_values)]
        print("*"* 20)
        print(f"step {noise_step}")
        print(f"topk_id_value_map {topk_id_value_map}")
        output_ids = outputs.sequences[:, input_token_len:]
        scores_sentence = torch.stack([score[out] for out, score in zip(output_ids.flatten(), scores_per_token)])
        log_prob = torch.sum(torch.log(scores_sentence))
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        step_to_probs_dicts.append({
            "steps" : noise_step,
            "token_probs" : topk_id_value_map
        })
    with open("results/logits/step_0_to_2000.json", "w") as f:
        json.dump(step_to_probs_dicts, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
