import copy

from vllm import LLM, SamplingParams
import json
import os
import argparse
from pathlib import Path


def read_jsonl(data_path):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(data_path, data):
    with open(data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data_path, data):
    with open(data_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Script")

    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature parameter for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p parameter for sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens for generation")
    parser.add_argument("--stop_tokens", type=str, default="</s>", help="Stop tokens for generation")
    parser.add_argument("--dataset_path", type=str, default="dataset/coco_instruct.json",
                        help="Path to the dataset")
    parser.add_argument("--prompt_structure", type=str,
                        default="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:",
                        help="Prefix for generation")

    parser.add_argument("--model_name_or_path", type=str, help="Path or name of the LLM model")
    parser.add_argument("--mode", type=str, default="description", help="Path or name of the LLM model")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output_result_path", type=str, default="output/coco", help="Path to output result file")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    stop_tokens = args.stop_tokens.split(",")

    data_list = read_json(args.dataset_path)
    if args.mode != "description":
        template = "I will give you a question and a response about an image. Suppose you are looking at a similar but different image with different details, such as more objects, less objects, or objects with different attributes. You need to provide the description for the other image. I will give you some examples and a new example, then you need to provide the new description.\n" \
                   "Examples:\n" \
                   "Question: Describe the image. Response: the image shows an old man walking across the street. There are many people on the sidewalks, and cars are running on the street.  Modified response: the image shows a woman running towards a car, there are no cars on the street.\n" \
                   "Question: What is the color of the apple? Response: the color of the apple is black. Modified response: the color of the apple is yellow.\n" \
                   "Question: What is on the left side of the boy? Response: there is a dog left to the boy. Modified response: it is a cat on the left of the boy.\n" \
                   "Question: How many people are there? Response: there are 3 people in the image. Modified response: there are only 1 person in the image.\n" \
                   "Question: How many oranges are in the bag? Response: 4. Modified response: 3.\n" \
                   "Question: what is the price of the knife? Response: The price shown in the image is 3 dollors. Modified response: The price shown in the image is 5 dollors.\n" \
                   "Question: {question} Response: {response}. Modified response:"
    else:
        template = "I will give you a description an image. You must modify the description by changing the original details, such as adding more object, deleting objects, or changing the attributes of objects (color, shape, size, location), make sure that the changed description is common and logical." \
                   "Description: {response}. Modified description:"
    prompts = []
    for instance in data_list:
        question = instance["prompt"]
        question = question.replace("<image>\n", "")
        question = question.replace("<image>", "")
        response = instance["completions"][0]["response"]
        # input = template.format(question=question, response=response)
        if args.mode == "description":
            input = template.format(response=response)
        else:
            input = template.format(question=question, response=response)
        prompts.append(args.prompt_structure.format(input=input))

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=1024, stop=stop_tokens)

    # Create an LLM.
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.gpu_num)

    outputs = llm.generate(prompts, sampling_params)

    results = copy.deepcopy(data_list)

    for data, output in zip(results, outputs):
        output_ = output.outputs[0].text
        data["completions"].append(
            {
                "score": 0,
                "response":output_[1:],
                "type":"llm-hal" #/"gt-rephrased"/"llm-hal"/"mllm-hal"
            }
        )

    os.makedirs(os.path.dirname(args.output_result_path), exist_ok=True)
    write_json(args.output_result_path + '/results.json', results)


if __name__ == "__main__":
    main()
