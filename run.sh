# replay

# CUDA_VISIBLE_DEVICES=0 python -m src.generate_instructions --config config.yaml \
#     --output data/replay/instructions.jsonl

# CUDA_VISIBLE_DEVICES=0 python -m src.generate_responses --config config.yaml \
#     --instructions data/replay/instructions.jsonl \
#     --output data/replay/candidates.jsonl

# CUDA_VISIBLE_DEVICES=0 python -m src.filter_responses --config config.yaml \
#     --candidates data/replay/candidates.jsonl \
#     --output data/replay/replay.jsonl

# public instructions


echo "Pulling public instructions..."
CUDA_VISIBLE_DEVICES=0 python -m src.pull_public_instructions --config config.yaml \
    --output data/public/instructions.jsonl

echo "Generating responses..."
CUDA_VISIBLE_DEVICES=0 python -m src.generate_responses --config config.yaml \
    --instructions data/public/instructions.jsonl \
    --output data/public/candidates.jsonl

echo "Filtering responses..."
CUDA_VISIBLE_DEVICES=0 python -m src.filter_responses --config config.yaml \
    --candidates data/public/candidates.jsonl \
    --output data/public/replay.jsonl

echo "Done!"

# --- Evaluation ---
# Run quick benchmark eval on the model
# CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --config eval/eval_config.yaml --suite quick

# Run full benchmark suite
# CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --config eval/eval_config.yaml --suite full

# Run specific tasks on a fine-tuned model
# CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --model /path/to/finetuned --tasks gsm8k,humaneval,ifeval

# Compare base vs fine-tuned results
# python -m eval.compare eval/results/Qwen3-4B-Instruct-2507_*.json eval/results/finetuned_*.json
