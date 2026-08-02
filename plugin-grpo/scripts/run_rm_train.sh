nnodes=1
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NNODES=$nnodes \
NODE_RANK=0 \
MASTER_PORT=8010 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type rm \
    --use_hf true \
    --model Skywork/Skywork-Reward-V2-Qwen3-1.7B \
    --tuner_type lora \
    --dataset /workspace/docker-dir-swift/rm_preference_train.jsonl \
    --val_dataset /workspace/docker-dir-swift/rm_preference_valid.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --lora_rank 256 \
    --lora_alpha 512 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 50 \
    --save_steps 300 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /workspace/docker-dir-swift/output/rm-skywork \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --deepspeed zero2 \
    --dataset_num_proc 20 \
    --center_rewards_coefficient 0.01
