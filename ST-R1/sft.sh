export WANDB_PROJECT=XXXX
export WANDB_NAME=XXX
mkdir -p /path/r1/ckpt/$WANDB_PROJECT/$WANDB_NAME

accelerate launch \
    --config_file scripts/zero3.yaml \
    src/open_r1_video/sft_video.py \
    --dataset_name /path/to/your/xxx.jsonl \
    --video_cache_dir /path/video_cache \
    --model_name_or_path /vi-lab/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --output_dir /path/r1/ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --bf16=True \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --optim="adamw_torch_fused" \
    --logging_steps=1 \
    --log_level debug \
    --log_level_replica debug \
    --save_strategy steps \
    --save_steps 2000 \
    --learning_rate 8e-5 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --push_to_hub False \
    --torch_dtype bfloat16 \
    --gradient_checkpointing True \
2>&1 | tee output_log.txt





