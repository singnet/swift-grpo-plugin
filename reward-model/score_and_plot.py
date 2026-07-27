import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer

postfix = "Prompt_Completion_1_5B"

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def compute_reward_score(model, tokenizer, prompt: str, response: str, device: str) -> float:
    """Compute reward score for a given prompt-response pair using Skywork Reward Model."""
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    
    # Remove duplicate bos_token if present
    if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
        conv_formatted = conv_formatted[len(tokenizer.bos_token):]

    conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        score = model(**conv_tokenized).logits[0][0].item()
    
    return score


def score_and_visualize(
    input_file: str = "parsed_data.jsonl",
    output_file: str = "scored_data.jsonl",
    model_name: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    device: str = "cuda:0",
    max_samples: int = None
):
    """
    Score all examples with Skywork Reward Model and create comparison plots
    between accuracy_reward = 1.0 and accuracy_reward = 0.0 using matplotlib only.
    """
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    
    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} examples. Loading model {model_name} on {device}...")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Computing reward scores...\n")

    for item in tqdm(data, desc="Scoring examples"):
        prompt = item.get("prompt", "") #f'{item.get("prompt", "")}\n\n{item.get("solution", "")}'
        completion = item.get("completion", "") #item.get("completion", "")
        
        try:
            score = compute_reward_score(model, tokenizer, prompt, completion, device)
            item["reward_score"] = round(score, 6)
        except Exception as e:
            print(f"Error processing example: {e}")
            item["reward_score"] = None

    # Save scored data
    save_jsonl(data, output_file)

    # =======================================================================
    # Visualization with matplotlib
    # =======================================================================
    print("\nGenerating plots with matplotlib...")

    # Prepare data for plotting
    scores_1 = [item["reward_score"] for item in data 
                if item.get("accuracy_reward") == 1.0 and item.get("reward_score") is not None]
    
    scores_0 = [item["reward_score"] for item in data 
                if item.get("accuracy_reward") == 0.0 and item.get("reward_score") is not None]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Boxplot comparison
    ax1.boxplot([scores_1, scores_0], patch_artist=True,
                boxprops=dict(facecolor="lightgreen", color="green"),
                medianprops=dict(color="darkgreen"),
                whiskerprops=dict(color="green"),
                capprops=dict(color="green"))
    
    ax1.set_xticklabels(["Accuracy = 1.0", "Accuracy = 0.0"])
    ax1.set_ylabel("Reward Score")
    ax1.set_title("Reward Score Distribution by Accuracy")
    ax1.grid(True, alpha=0.3)

    # 2. Histogram comparison
    ax2.hist(scores_1, bins=10, alpha=0.7, color="green", label="Accuracy = 1.0", edgecolor="black")
    ax2.hist(scores_0, bins=10, alpha=0.7, color="red", label="Accuracy = 0.0", edgecolor="black")
    
    ax2.set_xlabel("Reward Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Reward Score Histogram")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"reward_score_comparison_{postfix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    if scores_1:
        print(f"Accuracy = 1.0  |  Count: {len(scores_1):2d}  |  Mean Reward: {sum(scores_1)/len(scores_1):.4f}")
    else:
        print("No examples with Accuracy = 1.0")
        
    if scores_0:
        print(f"Accuracy = 0.0  |  Count: {len(scores_0):2d}  |  Mean Reward: {sum(scores_0)/len(scores_0):.4f}")
    else:
        print("No examples with Accuracy = 0.0")
    print(f"Plot saved as: {plot_path}")
    print("="*60)

    return data


if __name__ == "__main__":
    INPUT_FILE = "parsed_data.jsonl"
    OUTPUT_FILE = f"scored_data_{postfix}.jsonl"
    
    MAX_SAMPLES = None
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    score_and_visualize(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        model_name="Skywork/Skywork-Reward-V2-Qwen3-1.7B",
        device=DEVICE,
        max_samples=MAX_SAMPLES
    )
