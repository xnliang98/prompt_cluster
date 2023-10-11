import os
import sys

from tqdm.auto import tqdm
from utils import load_jsonl, save_jsonl


def convert_to_idtext(data):
    new_data = []
    for d in tqdm(data):
        idx = d['id']
        messages = d['messages']
        messages = [f"### {m['role']}\n{m['content']}" for m in messages]
        text = "\n\n".join(messages)
        new_data.append({
            "id": idx, 
            "text": text
        })
    return new_data


if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    if not os.path.exists(in_file):
        print("File is not found!")
        exit()
    
    data = load_jsonl(in_file)

    data = convert_to_idtext(data)
    save_jsonl(data, out_file)

