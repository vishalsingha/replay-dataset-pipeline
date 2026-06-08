"""
Benchmark evaluation runner using lm-evaluation-harness with vLLM backend.

Usage:
    python -m eval.run_eval --config eval/eval_config.yaml --suite quick
    python -m eval.run_eval --model /path/to/model --tasks gsm8k,ifeval
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_tasks(config: dict, suite: str | None, tasks_csv: str | None) -> list[str]:
    """Resolve which tasks to run from suite name or comma-separated task list."""
    if tasks_csv:
        return [t.strip() for t in tasks_csv.split(",")]
    if suite:
        suites = config.get("suites", {})
        if suite not in suites:
            available = ", ".join(suites.keys())
            print(f"Error: suite '{suite}' not found. Available: {available}")
            sys.exit(1)
        return suites[suite]
    return config.get("suites", {}).get("standard", [])


def build_lm_eval_args(config: dict, task_keys: list[str], model_path: str) -> dict:
    """Build kwargs for lm_eval.simple_evaluate()."""
    task_configs = config.get("tasks", {})
    lm_eval_tasks = []
    num_fewshot_map = {}
    needs_unsafe_code = False

    for key in task_keys:
        tc = task_configs.get(key, {})
        task_name = tc.get("task_name", key)
        lm_eval_tasks.append(task_name)
        if "num_fewshot" in tc:
            num_fewshot_map[task_name] = tc["num_fewshot"]
        if tc.get("confirm_unsafe_code", False):
            needs_unsafe_code = True

    # Use the minimum fewshot if all tasks share one, otherwise use per-task override
    common_fewshot = None
    fewshot_values = list(num_fewshot_map.values())
    if fewshot_values and len(set(fewshot_values)) == 1:
        common_fewshot = fewshot_values[0]

    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={config.get('tensor_parallel_size', 1)},"
        f"gpu_memory_utilization={config.get('gpu_memory_utilization', 0.9)},"
        f"max_model_len={config.get('max_model_len', 4096)},"
        f"dtype=auto,"
        f"seed={config.get('seed', 42)}"
    )

    seed = config.get("seed", 42)
    kwargs = {
        "model": "vllm",
        "model_args": model_args,
        "tasks": lm_eval_tasks,
        "batch_size": "auto",
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
        "log_samples": False,
    }

    if common_fewshot is not None:
        kwargs["num_fewshot"] = common_fewshot

    if needs_unsafe_code:
        kwargs["confirm_run_unsafe_code"] = True

    return kwargs


def extract_results_table(results: dict) -> list[dict]:
    """Extract per-task metrics into a flat list for display."""
    preferred_metrics = [
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "acc_norm,none",
        "acc_norm",
        "acc,none",
        "acc",
        "pass@1",
        "prompt_level_strict_acc,none",
        "prompt_level_strict_acc",
    ]
    skip_prefixes = ("alias", "name", "sample_len", "_stderr", "stderr")

    rows = []
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        primary_metric = None
        primary_value = None
        for preferred in preferred_metrics:
            if preferred in metrics and isinstance(metrics[preferred], (int, float)):
                primary_metric = preferred
                primary_value = metrics[preferred]
                break
        if primary_metric is None:
            for metric_key, value in metrics.items():
                if any(metric_key.startswith(s) or metric_key.endswith(s) for s in skip_prefixes):
                    continue
                if isinstance(value, (int, float)):
                    primary_metric = metric_key
                    primary_value = value
                    break
        rows.append({
            "task": task_name,
            "metric": primary_metric or "N/A",
            "score": primary_value if primary_value is not None else 0.0,
        })
    return rows


def print_summary(rows: list[dict], model_path: str) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {model_path}")
    print("=" * 60)
    print(f"  {'Task':<20} {'Metric':<30} {'Score':>8}")
    print("-" * 60)
    for row in sorted(rows, key=lambda r: r["task"]):
        score_str = f"{row['score'] * 100:.2f}%" if isinstance(row["score"], float) else str(row["score"])
        print(f"  {row['task']:<20} {row['metric']:<30} {score_str:>8}")
    print("=" * 60 + "\n")


def save_results(results: dict, rows: list[dict], model_path: str, output_dir: str, output_file: str | None = None) -> str:
    """Save results JSON and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        filename = output_file if output_file.endswith(".json") else f"{output_file}.json"
    else:
        model_name = Path(model_path).name.replace("/", "_")
        filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "model": model_path,
        "timestamp": timestamp,
        "summary": rows,
        "raw_results": results.get("results", {}),
        "config": results.get("config", {}),
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks via lm-evaluation-harness")
    parser.add_argument("--config", default="eval/eval_config.yaml", help="Path to eval config YAML")
    parser.add_argument("--model", default=None, help="HuggingFace model path (overrides config)")
    parser.add_argument("--suite", default=None, help="Benchmark suite: quick, standard, full")
    parser.add_argument("--tasks", default=None, help="Comma-separated task list (overrides --suite)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results (overrides config)")
    parser.add_argument("--output-file", default=None, help="Explicit output filename (placed in output-dir)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model or config.get("model_path", "")
    if not model_path:
        print("Error: no model_path specified in config or via --model flag")
        sys.exit(1)

    task_keys = resolve_tasks(config, args.suite, args.tasks)
    output_dir = args.output_dir or config.get("output_dir", "eval/results")

    print(f"Model: {model_path}")
    print(f"Tasks: {', '.join(task_keys)}")
    print(f"Output: {output_dir}")
    print()

    try:
        import lm_eval
    except ImportError:
        print("Error: lm-evaluation-harness not installed.")
        print("Install with: pip install 'lm-eval[vllm]'")
        sys.exit(1)

    eval_kwargs = build_lm_eval_args(config, task_keys, model_path)
    print("Starting evaluation...")
    results = lm_eval.simple_evaluate(**eval_kwargs)

    rows = extract_results_table(results)
    print_summary(rows, model_path)

    filepath = save_results(results, rows, model_path, output_dir, args.output_file)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
