deepspeed llava/train/bpo_llava_flash.py \
    --mm_projector_lr 2e-6 \
    --mm_projector_type mlp2x_gelu \
    --learning_rate 2e-6 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 256 \
    --model_name_or_path path-to-model \
    --version v1 \
    --data_path path-to-json-annotation-file \
    --image_folder path-to-image-folder \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir path-to-output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True  \
    --lora_enable

bash scripts/v1_5/eval/eval_multi_lora.sh path-to-model path-to-lora playground/data/eval/mm-vet.jsonl path-to-result path-to-images gpu-num temperature start_gpu
python scripts/convert_mmvet_for_eval.py --src path-to-result-jsonl --dst path-to-result-json
