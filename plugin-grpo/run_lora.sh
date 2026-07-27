nnodes=1
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_PORT=8010 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type grpo \
    --use_hf true \
    --model Qwen/Qwen2-0.5B-Instruct \
    --reward_funcs accuracy format \
    --reward_weights 0.7 0.3 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 4096 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset <path_to_train_dataset> \
    --val_dataset <path_to_val_dataset> \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system '/home/ms-swift/examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear
