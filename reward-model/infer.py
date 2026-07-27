import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "output/skywork-reward-mmlu"
device = "cuda:0"

rm = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_reward(prompt, response):
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        score = rm(**inputs).logits[0][0].item()
    return score