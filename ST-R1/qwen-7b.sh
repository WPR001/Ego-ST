export WANDB_PROJECT=Qwen2-VL-7B-Video-GRPO
export WANDB_NAME=video-r1-without12

mkdir -p /vi-lab/Ego-ST-bench/r1/ckpt/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1_video/grpo.py \
    --deepspeed scripts/zero3.json \
    --output_dir /vi-lab/Ego-ST-bench/r1/ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path /vi-lab/Ego-ST-bench/r1/ckpt/Qwen2-VL-7B-Video-GRPO/merge_cot_mcq_without12 \
    --dataset_name st_bench \
    --jsonl_path /vi-lab/Ego-ST-bench/ACM_bench/ablation_domain/train/mcq_without_part12_630.jsonl \
    --max_prompt_length 4096 \
    --learning_rate 1e-5 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 1000 \
    --save_only_model true
2>&1 | tee output_log.txt
