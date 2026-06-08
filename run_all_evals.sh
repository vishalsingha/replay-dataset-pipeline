#!/bin/bash
# Evaluate all fine-tuned models + base model on benchmark suite
# Models are stored in /home/azureuser/nvme_0/replay/

set -e

EVAL_CMD="python -m eval.run_eval --config eval/eval_config.yaml"
SUITE="full"  # change to "standard" or "full" for more comprehensive eval
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_ROOT="/home/azureuser/nvme_0/replay"

export HF_ALLOW_CODE_EVAL="1"

echo "=========================================="
echo "  Benchmark Evaluation - All Models"
echo "  Suite: ${SUITE}"
echo "=========================================="

# 1. Base model (reference)
echo ""
echo "[1/5] Evaluating base model: ${BASE_MODEL}"
CUDA_VISIBLE_DEVICES=0 $EVAL_CMD --model "$BASE_MODEL" --suite "$SUITE" \
    --output-file "base_qwen3_4b.json"

# 2. task_only/saved_model
echo ""
echo "[2/5] Evaluating: task_only/saved_model"
CUDA_VISIBLE_DEVICES=0 $EVAL_CMD --model "${MODEL_ROOT}/task_only/saved_model" --suite "$SUITE" \
    --output-file "task_only.json"

# 3. replay_task/saved_model
echo ""
echo "[3/5] Evaluating: replay_task/saved_model"
CUDA_VISIBLE_DEVICES=0 $EVAL_CMD --model "${MODEL_ROOT}/replay_task/saved_model" --suite "$SUITE" \
    --output-file "replay_task.json"

# 4. public_replay_task/saved_model
echo ""
echo "[4/5] Evaluating: public_replay_task/saved_model"
CUDA_VISIBLE_DEVICES=0 $EVAL_CMD --model "${MODEL_ROOT}/public_replay_task/saved_model" --suite "$SUITE" \
    --output-file "public_replay_task.json"

# 5. replay_public_replay_task (no saved_model, use latest checkpoint)
echo ""
echo "[5/5] Evaluating: replay_public_replay_task/checkpoint-60"
CUDA_VISIBLE_DEVICES=0 $EVAL_CMD --model "${MODEL_ROOT}/replay_public_replay_task/checkpoint-60" --suite "$SUITE" \
    --output-file "replay_public_replay_task.json"

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "  Results saved in: eval/results/"
echo "=========================================="
echo ""
echo "To compare models against base:"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/task_only.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/replay_task.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/public_replay_task.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/replay_public_replay_task.json"
