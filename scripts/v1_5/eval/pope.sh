#!/bin/bash
#--model-base ../pretrained_weights/llava1.5_7b \
#--model-path ./checkpoints/dpo/llava1.5_7b-lora32-lr2e-6-lrv_5w-1e/ \
python -m llava.eval.model_vqa_loader \
    --model-path ../pretrained_weights/llava1.5_7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ../data/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/dpo/llava1.5_7b-lora32-lr2e-6-lrv_5w-1e.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava1.5_7b-lora32-lr2e-6-lrv_5w-1e.jsonl
