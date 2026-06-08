"""
DS-1000 evaluation: data science code generation benchmark.

DS-1000 contains 1000 data science problems spanning 7 Python libraries
(NumPy, Pandas, SciPy, Scikit-learn, TensorFlow, PyTorch, Matplotlib).
Uses functional unit tests and surface-form constraints.

This script generates completions via vLLM and evaluates using execution.

Usage:
    python -m eval.ds1000_run --model Qwen/Qwen3-4B-Instruct-2507
    python -m eval.ds1000_run --model /path/to/finetuned --output-file ds1000_finetuned.json

Requirements:
    pip install ds1000 vllm
    # OR use bigcode-evaluation-harness:
    # pip install bigcode-eval-harness
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def generate_and_evaluate(model_path: str, tp_size: int = 1, mode: str = "insertion") -> dict:
    """Generate DS-1000 solutions using vLLM and evaluate."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("Loading DS-1000 dataset...")
    ds = load_dataset("xlangai/DS-1000", split="test")
    print(f"Loaded {len(ds)} problems")

    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(temperature=0.0, max_tokens=1024)

    prompts = []
    problems = []
    for example in ds:
        prompt_text = example.get("prompt", "")
        problems.append(example)
        messages = [
            {"role": "system", "content": "You are an expert Python data scientist. Complete the code to solve the given problem. Output ONLY the code, no explanations."},
            {"role": "user", "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)

    print(f"Generating solutions for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling)

    correct = 0
    total = 0
    results_per_lib = {}

    for problem, output in zip(problems, outputs):
        response = output.outputs[0].text.strip()
        lib = problem.get("metadata", {}).get("library", problem.get("lib", "unknown"))
        if lib not in results_per_lib:
            results_per_lib[lib] = {"correct": 0, "total": 0}
        results_per_lib[lib]["total"] += 1
        total += 1

        # Execute and check
        test_code = problem.get("code_context", "") + "\n" + response + "\n" + problem.get("test", "")
        try:
            exec_globals = {}
            exec(test_code, exec_globals)
            correct += 1
            results_per_lib[lib]["correct"] += 1
        except Exception:
            pass

    accuracy = correct / total if total > 0 else 0

    summary = {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_library": {
            lib: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for lib, v in results_per_lib.items()
        },
    }

    print(f"\n{'=' * 50}")
    print(f"  DS-1000 Results: {Path(model_path).name}")
    print(f"{'=' * 50}")
    print(f"  Overall: {correct}/{total} ({accuracy*100:.2f}%)")
    print(f"  {'Library':<15} {'Accuracy':>10} {'Count':>8}")
    print(f"{'-' * 50}")
    for lib, v in sorted(summary["per_library"].items()):
        print(f"  {lib:<15} {v['accuracy']*100:>9.2f}% {v['total']:>8}")
    print(f"{'=' * 50}\n")

    return summary


def save_results(model_path: str, summary: dict, output_dir: str, output_file: str | None) -> str:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        filename = output_file if output_file.endswith(".json") else f"{output_file}.json"
    else:
        model_name = Path(model_path).name.replace("/", "_")
        filename = f"ds1000_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "benchmark": "ds1000",
        "model": model_path,
        "timestamp": timestamp,
        "summary": summary,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run DS-1000 evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--output-dir", default="eval/results", help="Output directory")
    parser.add_argument("--output-file", default=None, help="Output filename")
    args = parser.parse_args()

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    summary = generate_and_evaluate(args.model, args.tp_size)
    save_results(args.model, summary, args.output_dir, args.output_file)


if __name__ == "__main__":
    main()
