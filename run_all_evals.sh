#!/bin/bash
# Evaluate all fine-tuned models + base model on benchmark suite
# Models are stored in /home/azureuser/nvme_0/replay/

set -e

# Load API keys from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

EVAL_CMD="python -m eval.run_eval --config eval/eval_config.yaml"
SUITE="full"  # change to "quick", "standard", "full", or "safety"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_ROOT="/home/azureuser/nvme_0/replay_lora"

export HF_ALLOW_CODE_EVAL="1"
export VLLM_RPC_TIMEOUT="300000"
export CUDA_VISIBLE_DEVICES="0"

echo "=========================================="
echo "  Benchmark Evaluation - All Models"
echo "  Suite: ${SUITE}"
echo "=========================================="

# 1. Base model (reference)
# echo ""
# echo "[1/6] Evaluating base model: ${BASE_MODEL}"
# $EVAL_CMD --model "$BASE_MODEL" --suite "$SUITE" \
#     --output-file "base_qwen3_4b.json"

# 2. task_only/saved_full_model
echo ""
echo "[2/6] Evaluating: task_only/saved_full_model"
$EVAL_CMD --model "${MODEL_ROOT}/task_only/saved_full_model" --suite "$SUITE" \
    --output-file "task_only.json"

# 3. replay_task/saved_full_model
echo ""
echo "[3/6] Evaluating: replay_task/saved_full_model"
$EVAL_CMD --model "${MODEL_ROOT}/replay_task/saved_full_model" --suite "$SUITE" \
    --output-file "replay_task.json"

# 4. public_replay_task/saved_full_model
echo ""
echo "[4/6] Evaluating: public_replay_task/saved_full_model"
$EVAL_CMD --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --suite "$SUITE" \
    --output-file "public_replay_task.json"

# 5. replay_public_replay_task/saved_full_model
echo ""
echo "[5/6] Evaluating: replay_public_replay_task/saved_full_model"
$EVAL_CMD --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --suite "$SUITE" \
    --output-file "replay_public_replay_task.json"

# 6. public_conv_task/saved_full_model
echo ""
echo "[6/6] Evaluating: public_conv_task/saved_full_model"
$EVAL_CMD --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --suite "$SUITE" \
    --output-file "public_conv_task.json"

echo ""
echo "=========================================="
echo "  All lm-eval evaluations complete!"
echo "  Results saved in: eval/results/"
echo "=========================================="
echo ""
echo "To compare models against base:"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/task_only.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/replay_task.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/public_replay_task.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/replay_public_replay_task.json"
echo "  python -m eval.compare eval/results/base_qwen3_4b.json eval/results/public_conv_task.json"

# --- Custom Benchmarks ---

echo ""
echo "=========================================="
echo "  Custom Benchmarks (LLM-as-judge + Code)"
echo "=========================================="

# Safety benchmarks (HaluEval + JailbreakBench) — no API key needed
echo ""
echo "--- Safety Benchmarks (HaluEval + JailbreakBench) ---"
python -m eval.safety_bench --tp-size 1 --model "$BASE_MODEL" --bench all --output-file "safety_base.json"
python -m eval.safety_bench --tp-size 1 --model "${MODEL_ROOT}/task_only/saved_full_model" --bench all --output-file "safety_task_only.json"
python -m eval.safety_bench --tp-size 1 --model "${MODEL_ROOT}/replay_task/saved_full_model" --bench all --output-file "safety_replay_task.json"
python -m eval.safety_bench --tp-size 1 --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --bench all --output-file "safety_public_replay_task.json"
python -m eval.safety_bench --tp-size 1 --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --bench all --output-file "safety_replay_public_replay_task.json"
python -m eval.safety_bench --tp-size 1 --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --bench all --output-file "safety_public_conv_task.json"

# MT-Bench (requires Azure OpenAI / OPENAI_API_KEY)
echo ""
echo "--- MT-Bench ---"
python -m eval.mt_bench --tp-size 1 --model "$BASE_MODEL" --output-file "mt_bench_base.json"
python -m eval.mt_bench --tp-size 1 --model "${MODEL_ROOT}/task_only/saved_full_model" --output-file "mt_bench_task_only.json"
python -m eval.mt_bench --tp-size 1 --model "${MODEL_ROOT}/replay_task/saved_full_model" --output-file "mt_bench_replay_task.json"
python -m eval.mt_bench --tp-size 1 --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --output-file "mt_bench_public_replay_task.json"
python -m eval.mt_bench --tp-size 1 --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --output-file "mt_bench_replay_public_replay_task.json"
python -m eval.mt_bench --tp-size 1 --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --output-file "mt_bench_public_conv_task.json"

# AlpacaEval 2.0 (requires Azure OpenAI / OPENAI_API_KEY)
# echo ""
# echo "--- AlpacaEval 2.0 ---"
# python -m eval.alpaca_eval_run --tp-size 1 --model "$BASE_MODEL" --output-file "alpaca_base.json"
# python -m eval.alpaca_eval_run --tp-size 1 --model "${MODEL_ROOT}/task_only/saved_full_model" --output-file "alpaca_task_only.json"
# python -m eval.alpaca_eval_run --tp-size 1 --model "${MODEL_ROOT}/replay_task/saved_full_model" --output-file "alpaca_replay_task.json"
# python -m eval.alpaca_eval_run --tp-size 1 --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --output-file "alpaca_public_replay_task.json"
# python -m eval.alpaca_eval_run --tp-size 1 --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --output-file "alpaca_replay_public_replay_task.json"
# python -m eval.alpaca_eval_run --tp-size 1 --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --output-file "alpaca_public_conv_task.json"

# LiveCodeBench (requires LiveCodeBench repo cloned)
# echo ""
# echo "--- LiveCodeBench ---"
# python -m eval.livecodebench_run --tp-size 1 --model "$BASE_MODEL" --output-file "lcb_base.json"
# python -m eval.livecodebench_run --tp-size 1 --model "${MODEL_ROOT}/task_only/saved_full_model" --output-file "lcb_task_only.json"
# python -m eval.livecodebench_run --tp-size 1 --model "${MODEL_ROOT}/replay_task/saved_full_model" --output-file "lcb_replay_task.json"
# python -m eval.livecodebench_run --tp-size 1 --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --output-file "lcb_public_replay_task.json"
# python -m eval.livecodebench_run --tp-size 1 --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --output-file "lcb_replay_public_replay_task.json"
# python -m eval.livecodebench_run --tp-size 1 --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --output-file "lcb_public_conv_task.json"

# DS-1000 (data science code)
# echo ""
# echo "--- DS-1000 ---"
# python -m eval.ds1000_run --tp-size 1 --model "$BASE_MODEL" --output-file "ds1000_base.json"
# python -m eval.ds1000_run --tp-size 1 --model "${MODEL_ROOT}/task_only/saved_full_model" --output-file "ds1000_task_only.json"
# python -m eval.ds1000_run --tp-size 1 --model "${MODEL_ROOT}/replay_task/saved_full_model" --output-file "ds1000_replay_task.json"
# python -m eval.ds1000_run --tp-size 1 --model "${MODEL_ROOT}/public_replay_task/saved_full_model" --output-file "ds1000_public_replay_task.json"
# python -m eval.ds1000_run --tp-size 1 --model "${MODEL_ROOT}/replay_public_replay_task/saved_full_model" --output-file "ds1000_replay_public_replay_task.json"
# python -m eval.ds1000_run --tp-size 1 --model "${MODEL_ROOT}/public_conv_task/saved_full_model" --output-file "ds1000_public_conv_task.json"
