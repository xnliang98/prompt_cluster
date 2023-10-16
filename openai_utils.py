import tiktoken
import openai
import os
import json
import argparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import random
import time
from tqdm import tqdm

from utils import load_jsonl



# TEXT_BOS = "<s>"
# TEXT_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
TEXT_PROMPT = ""
TEXT_REPLACE_TABLE = {"<|end_of_turn|>": "\n\n"}

# API

MAX_TOKENS = 8191
# BATCH_SIZE = 64

MODEL_TYPE = "text-embedding-ada-002"
MODEL_TOKENIZER = tiktoken.encoding_for_model(MODEL_TYPE)

def save_jsonl(data, path):
    with open(path, "a+", encoding="utf-8") as f:
        for line in data:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def preprocess_text(text: str):
    # Preprocess text, remove bos, add prompt and replace
    # if text.startswith(TEXT_BOS):
    #     text = text[len(TEXT_BOS):]

    for src, dst in TEXT_REPLACE_TABLE.items():
        text = text.replace(src, dst)

    text = TEXT_PROMPT + text

    # Tokenize and truncate
    tokens = MODEL_TOKENIZER.encode(text, disallowed_special=())
    tokens = tokens[:MAX_TOKENS]
    return tokens

def calculate_embeddings(samples):

    tokens_chunk = list(map(preprocess_text, samples))
    # Call API
    # Reference: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb
    response = embedding_with_backoff(model=MODEL_TYPE, input=tokens_chunk)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input

    embeddings_chunk = [e["embedding"] for e in response["data"]]

    return embeddings_chunk

def batchize(data, batch_size=2):
    idx, texts = [], []
    for d in data:
        idx.append(d['id'])
        texts.append(d['text'])
    batched_text = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    idx = [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
    return idx, batched_text

def get_ada_embeddings(data, batch_size=64, path=None):
    idxes, batched_text = batchize(data, batch_size)

    embeddings = []

    for idx, sentences in tqdm(zip(idxes, batched_text), total=len(idxes)):
        try:
            emb = calculate_embeddings(sentences)
            # embeddings.extend(emb)
            id_embeddings = []
            for ix, embedding in zip(idx, emb):
                id_embeddings.append({
                    "id": ix,
                    "embedding": embedding
                })
            save_jsonl(id_embeddings, path)
            time.sleep(6)
        except Exception as ex:
            print(ex)
            break

    # assert len(idxes) == len(embeddings), "id长度和embedding长度不一致，请检查embedding获取是否合理！"

    # id_embeddings = []
    # for idx, embedding in zip(idxes, embeddings):
    #     id_embeddings.append({
    #         "id": idx,
    #         "embedding": embedding.tolist()
    #     })
    # save_jsonl(id_embeddings, path)
    # return id_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", default=None, help="输入文件所在文件夹")
    parser.add_argument("--in_file", default=None, help="输入文件名，{'id': xxx, 'text': xxx}")
    parser.add_argument("--out_dir", default=None, help="向量输出位置")
    parser.add_argument("--emb_method", default="bert", help="表征获取方法")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--reduce_dim", default=3, type=int)
    parser.add_argument("--cluster_algo", default="KMeans", help="建议使用：KMeans、DBSCAN、BisectingKMeans")

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    in_file_name = args.in_file.replace(".jsonl", "")
    out_file_name = f"{in_file_name}.{args.cluster_algo}.labels.tsv"
    out_emb_name = f"{in_file_name}.emb.jsonl"
    out_reduced_dim_name = f"{in_file_name}.emb.{str(args.reduce_dim)}.jsonl"

    # 加载原始文本数据
    data = load_jsonl(os.path.join(args.in_dir, args.in_file))
    
    
    if os.path.exists(os.path.join(args.out_dir, out_emb_name)):
        id_embeddings = load_jsonl(os.path.join(args.out_dir, out_emb_name))
        saved_cnt = len(id_embeddings)
        data = data[saved_cnt:]
        id_embeddings = get_ada_embeddings(data, args.batch_size, os.path.join(args.out_dir, out_emb_name))
    else:
        id_embeddings = get_ada_embeddings(data, args.batch_size, os.path.join(args.out_dir, out_emb_name))
        # save_jsonl(id_embeddings, os.path.join(args.out_dir, out_emb_name))