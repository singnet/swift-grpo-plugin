import json

def clean_prompt_prefix(input_file="parsed_data.jsonl", output_file="parsed_data_cleaned.jsonl"):
    """
    Removes the long system prefix from the beginning of each prompt.
    """
    prefix = "<｜begin▁of▁sentence｜>A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\n"
    
    cleaned_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            item = json.loads(line)
            total_count += 1
            
            prompt = item.get("prompt", "")
            
            # Remove the prefix if it exists at the beginning
            if prompt.startswith(prefix):
                item["prompt"] = prompt[len(prefix):].strip()
                cleaned_count += 1
            else:
                # Try partial match in case of slight variations
                if "The assistant first thinks about the reasoning process" in prompt:
                    # Find and remove the long instruction
                    start_pos = prompt.find("The user asks a question")
                    if start_pos != -1:
                        item["prompt"] = prompt[start_pos:].strip()
                        cleaned_count += 1
            
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processing completed!")
    print(f"Total samples: {total_count}")
    print(f"Cleaned prompts: {cleaned_count}")
    print(f"Cleaned file saved as: {output_file}")


if __name__ == "__main__":
    clean_prompt_prefix()
