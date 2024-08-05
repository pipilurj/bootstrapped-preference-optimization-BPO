# Data Generation for BPO
- For image weakening, we are inspired by [VCD](https://github.com/DAMO-NLP-SG/VCD) to add noise to image features, which generates negative responses with pretraining bias.
- For error injection, we utilize the pretrained LLM that is the same as the base model of the MLLM to directly inject erroneous concepts.


## Install environments
For error injection, please install VLLM to speed up inference.
```
pip intall vllm
```
If it can't run, please consider building the package from source.

## Image weakening
```bash
cd data_generation
python run_llava_image_weakening.py --model-path liuhaotian/llava-v1.5-13b --image_file YOUR_IMAGE_PATH --query YOUR_JSON_PATH --save_path OUTPUT_PATH
```

`YOUR_JSON_PATH` should be a list of json file and has the following format
```
[
    {'prompt': 'What do you see happening in this image?\n<image>',
        'image': 'coco/train2017/000000000009.jpg',
        'completions': [{'score': 1,
        'response': "xxxx",
        'type': 'gt'}]},

    {'prompt': 'question\n<image>',
        'image': 'coco/train2017/0000000000010.jpg',
        'completions': [{'score': 1,
        'response': "xxxx",
        'type': 'gt'}]},
    .......
]
```
Note: Image weakening currently runs on generic inference pipeline. We will consider integrating it with MLLM acceleration framework, e.g,. https://github.com/InternLM/lmdeploy.

## Error injection
```bash
cd data_generation
python error_injection.py --model_name_or_path PATH-TO-LLM --dataset_path PATH-TO-SFT-DATA --output_result_path PATH-TO-PREFERENCE-DATA
```

