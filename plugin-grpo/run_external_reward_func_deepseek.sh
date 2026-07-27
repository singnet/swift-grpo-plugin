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
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reward_funcs accuracy_openr1_reward tag_count_openr1_reward format \
    --reward_weights 0.7 0.05 0.25 \
    --external_plugins /home/swift-grpo-plugin/plugin-grpo/reward_plugin.py \
    --use_vllm true \
    --tuner_type lora \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 4096 \
    --torch_dtype bfloat16 \
    --dataset <path_to_train_dataset> \
    --val_dataset <path_to_val_dataset> \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 25 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 2 \
    --temperature 0.9 \
    --system '/home/ms-swift/examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --offload_model true \
    --offload_optimizer true \
    --loss_type grpo \
    --vllm_enable_lora false \
    --advantage_estimator grpo \
    --lora_rank 256 \
    --lora_alpha 512
