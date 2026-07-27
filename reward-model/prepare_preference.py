import json

def prepare_ms_swift_rlhf(input_file="parsed_data_cleaned.jsonl", output_file="rlhf_dataset.jsonl"):
    """
    Creates dataset in the format expected by swift rlhf.
    """
    data = []
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            prompt = item.get("prompt", "").strip()
            completion = item.get("completion", "").strip()
            accuracy = item.get("accuracy_reward")
            
            if len(prompt) < 100 or len(completion) < 50 or accuracy is None:
                continue
            
            # Create conversation format
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            
            entry = {
                "messages": messages,
                "rejected_messages": None,   # will be filled for pairs
            }
            
            # For rlhf we need pairs, so we skip single samples for now
            # Better to create proper pairs later
            data.append(entry)
            count += 1
            if count >= 300:  # limit for testing
                break
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {len(data)} samples → {output_file}")
    return data


if __name__ == "__main__":
    prepare_ms_swift_rlhf()
