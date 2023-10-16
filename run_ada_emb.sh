# export OPENAI_API_KEY=""

python3 openai_utils.py \
    --in_dir data/ \
    --in_file sharegpt-zh.jsonl \
    --out_dir output_ada/ \
    --cluster_algo KMeans \
    --emb_method ada