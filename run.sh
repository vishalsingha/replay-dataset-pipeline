CUDA_VISIBLE_DEVICES=0 python -m src.generate_instructions --config config.yaml \
    --instructions data/replay/instructions.jsonl \
    --output data/replay/instructions.jsonl

CUDA_VISIBLE_DEVICES=0 python -m src.generate_responses --config config.yaml \
    --instructions data/replay/instructions.jsonl \
    --output data/replay/candidates.jsonl

CUDA_VISIBLE_DEVICES=0 python -m src.filter_responses --config config.yaml \
    --candidates data/replay/candidates.jsonl \
    --output data/replay/replay.jsonl


# CUDA_VISIBLE_DEVICES=0 python -m src.generate_responses --config config.yaml \
#     --instructions data/public_instructions/instructions.jsonl \
#     --output data/public_instructions/candidates.jsonl


# CUDA_VISIBLE_DEVICES=0 python -m src.filter_responses --config config.yaml \
#     --candidates data/public_instructions/candidates.jsonl \
#     --output data/public_instructions/replay.jsonl

