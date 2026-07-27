# Reward Model for MS-Swift GRPO

## Requirements

- GPU with at least 24 GB of VRAM
- CUDA 12.1+
- Python 3.10+
- ms-swift 4.x

## Set up

Configure docker for ms-swift.

## Data preparation

**The source file**

parsed_data.json - Contains fields:
- prompt
- completion
- accuracy_reward (1.0 = correct, 0.0 = incorrect)



## Train Reward Model

Instructions for fine-tuning the **Skywork-Reward-V2-Qwen3-1.7B** model based on preference data using **ms-swift**.

