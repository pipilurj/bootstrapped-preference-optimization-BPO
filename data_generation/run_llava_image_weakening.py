import argparse
import torch
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from glob import glob
import json
from image_weakening_utils.add_noise import add_diffusion_noise
from PIL import Image
from image_weakening_utils.sample import evolve_vcd_sampling
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
evolve_vcd_sampling()

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    import json
    with open(args.query) as f:
        ds = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, mode="bpo")
    with open(args.save_path,'a') as f:
        for i in tqdm(range(len(ds))):

            qs = ds[i]['prompt']
            if '<image>\n' in qs:
                qs = qs.replace('<image>\n','')
            if 'n<image>' in qs:
                qs=qs.replace('n<image>','')
            if '<image>' in qs:
                qs=qs.replace('<image>','')
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # prompt="Describe the image in detals."
            # print(prompt)
            print(os.path.join(args.image_file ,ds[i]['image']))
            image = load_image(os.path.join(args.image_file ,ds[i]['image']))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    # images_cd=(image_tensor_cd if image_tensor_cd is not None else None),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                    pad_token_id=tokenizer.eos_token_id,  # Pad token
                    use_cache=True,
                    images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                    cd_alpha = args.cd_alpha,
                    cd_beta = args.cd_beta,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ds[i]['completions'].append({'score':0, 'response': outputs,'type':'mllm-hal'})

            json.dump(ds[image],f)
            f.write('\n')
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--noise_step", type=int, default=900)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)
