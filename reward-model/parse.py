import json
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_answer_from_completion(completion: str) -> str:
    """Extracts the final answer from completion."""
    # search <answer> ... </answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # search \boxed{...}
    boxed_match = re.search(r'\\boxed\{(.*?)\}', completion, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # search A/B/C/D
    option_match = re.search(r'(A|B|C|D)\.', completion)
    if option_match:
        return option_match.group(1)
    
    return completion[:500]  # if nothing is found, the first 500 characters


def parse_completions_jsonl(file_path: str, output_file: str = "parsed_data.jsonl"):
    """
    Parses the completions.jsonl file and saves structured data.
    """
    data_list: List[Dict[str, Any]] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                
                # Process each sample
                num_samples = len(item.get("prompt", []))
                
                for i in range(num_samples):
                    parsed = {
                        "step": item.get("step", [])[i] if i < len(item.get("step", [])) else None,
                        "prompt": clean_text(item.get("prompt", [])[i]) if i < len(item.get("prompt", [])) else "",
                        "completion": clean_text(item.get("completion", [])[i]) if i < len(item.get("completion", [])) else "",
                        "extracted_answer": extract_answer_from_completion(
                            item.get("completion", [])[i] if i < len(item.get("completion", [])) else ""
                        ),
                        "solution": item.get("solution", [])[i] if i < len(item.get("solution", [])) else None,
                        "accuracy_reward": float(item.get("AccuracyReward", [])[i]) if i < len(item.get("AccuracyReward", [])) else None,
                        "tag_count_reward": float(item.get("TagCountReward", [])[i]) if i < len(item.get("TagCountReward", [])) else None,
                        "format_reward": float(item.get("Format", [])[i]) if i < len(item.get("Format", [])) else None,
                        "advantage": float(item.get("advantages", [])[i]) if i < len(item.get("advantages", [])) else None,
                    }
                    data_list.append(parsed)
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    # Saving the result
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Parsing is complete! {len(data_list)} samples have been processed.")
    print(f"The result is saved in {output_file}")
    
    # Brief statistics
    correct = sum(1 for d in data_list if d["accuracy_reward"] == 1.0)
    print(f"Accuracy (AccuracyReward == 1.0): {correct}/{len(data_list)} ({correct/len(data_list)*100:.1f}%)")
    
    return data_list


if __name__ == "__main__":
    input_file = "completions.jsonl"
    parse_completions_jsonl(input_file)
