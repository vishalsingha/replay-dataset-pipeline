"""
Compare evaluation results between two models side-by-side.

Useful for measuring catastrophic forgetting: compare base model vs fine-tuned model.

Usage:
    python -m eval.compare results_base.json results_finetuned.json
    python -m eval.compare eval/results/Qwen3*.json eval/results/finetuned*.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path


def load_result(path: str) -> dict:
    """Load a single evaluation result JSON."""
    with open(path) as f:
        return json.load(f)


def resolve_path(pattern: str) -> str:
    """Resolve a path that might be a glob pattern to the most recent match."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"Error: no files matching '{pattern}'")
        sys.exit(1)
    return matches[-1]  # most recent by filename (timestamp-sorted)


PREFERRED_METRICS = [
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
SKIP_SUFFIXES = ("_stderr", "stderr")
SKIP_KEYS = ("alias", "name", "sample_len")


def _extract_primary_metric(metrics: dict) -> tuple[str | None, float | None]:
    """Pick the best metric from a task's result dict."""
    for preferred in PREFERRED_METRICS:
        if preferred in metrics and isinstance(metrics[preferred], (int, float)):
            return preferred, metrics[preferred]
    for key, value in metrics.items():
        if any(key.startswith(s) or key.endswith(s) for s in SKIP_SUFFIXES):
            continue
        if key in SKIP_KEYS:
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def build_score_map(result: dict) -> dict[str, dict]:
    """Build task -> {metric, score} map from result JSON."""
    score_map = {}
    # Prefer extracting from raw_results for accurate metric selection
    raw = result.get("raw_results", {})
    if raw:
        for task_name, metrics in raw.items():
            metric, score = _extract_primary_metric(metrics)
            if metric is not None:
                score_map[task_name] = {"metric": metric, "score": score}
    else:
        for entry in result.get("summary", []):
            score_map[entry["task"]] = {
                "metric": entry["metric"],
                "score": entry["score"],
            }
    return score_map


def print_comparison(
    model_a: str,
    model_b: str,
    scores_a: dict[str, dict],
    scores_b: dict[str, dict],
) -> None:
    """Print a side-by-side comparison table with delta highlighting."""
    all_tasks = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))

    col_a = Path(model_a).name[:25]
    col_b = Path(model_b).name[:25]

    print("\n" + "=" * 80)
    print("  Model Comparison")
    print(f"  A: {model_a}")
    print(f"  B: {model_b}")
    print("=" * 80)
    print(f"  {'Task':<18} {'Metric':<22} {col_a:>10} {col_b:>10} {'Delta':>10}")
    print("-" * 80)

    regressions = 0
    improvements = 0

    for task in all_tasks:
        sa = scores_a.get(task, {})
        sb = scores_b.get(task, {})
        score_a = sa.get("score", None)
        score_b = sb.get("score", None)
        metric = sa.get("metric") or sb.get("metric", "N/A")

        str_a = f"{score_a * 100:.2f}%" if score_a is not None else "—"
        str_b = f"{score_b * 100:.2f}%" if score_b is not None else "—"

        if score_a is not None and score_b is not None:
            delta = score_b - score_a
            delta_pct = delta * 100
            if delta > 0.001:
                marker = f"+{delta_pct:.2f}%"
                improvements += 1
            elif delta < -0.001:
                marker = f"{delta_pct:.2f}% <<<"
                regressions += 1
            else:
                marker = "="
        else:
            marker = "—"

        print(f"  {task:<18} {metric:<22} {str_a:>10} {str_b:>10} {marker:>10}")

    print("-" * 80)
    print(f"  Summary: {improvements} improvements, {regressions} regressions")
    if regressions > 0:
        print("  <<< marks regressions (potential catastrophic forgetting)")
    print("=" * 80 + "\n")


def save_comparison(
    model_a: str,
    model_b: str,
    scores_a: dict[str, dict],
    scores_b: dict[str, dict],
    output_path: str,
) -> None:
    """Save comparison as JSON."""
    all_tasks = sorted(set(list(scores_a.keys()) + list(scores_b.keys())))
    comparison = []
    for task in all_tasks:
        sa = scores_a.get(task, {})
        sb = scores_b.get(task, {})
        score_a = sa.get("score")
        score_b = sb.get("score")
        delta = (score_b - score_a) if (score_a is not None and score_b is not None) else None
        comparison.append({
            "task": task,
            "metric": sa.get("metric") or sb.get("metric"),
            "score_a": score_a,
            "score_b": score_b,
            "delta": delta,
        })

    output = {
        "model_a": model_a,
        "model_b": model_b,
        "comparison": comparison,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between two models"
    )
    parser.add_argument("result_a", help="Path (or glob) to model A results JSON (baseline)")
    parser.add_argument("result_b", help="Path (or glob) to model B results JSON (fine-tuned)")
    parser.add_argument("--output", default=None, help="Save comparison JSON to this path")
    args = parser.parse_args()

    path_a = resolve_path(args.result_a)
    path_b = resolve_path(args.result_b)

    print(f"Loading: {path_a}")
    print(f"Loading: {path_b}")

    result_a = load_result(path_a)
    result_b = load_result(path_b)

    model_a = result_a.get("model", path_a)
    model_b = result_b.get("model", path_b)

    scores_a = build_score_map(result_a)
    scores_b = build_score_map(result_b)

    print_comparison(model_a, model_b, scores_a, scores_b)

    if args.output:
        save_comparison(model_a, model_b, scores_a, scores_b, args.output)


if __name__ == "__main__":
    main()
