import json


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n")

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
