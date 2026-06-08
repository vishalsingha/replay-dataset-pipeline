"""
LiveCodeBench evaluation: contamination-free code generation benchmark.

LiveCodeBench uses continuously updated competitive programming problems
(post-2023) ensuring no data contamination. Evaluates pass@1 and pass@5.

This script wraps the LiveCodeBench repo's runner. It requires the
LiveCodeBench repository to be cloned separately.

Usage:
    # First-time setup:
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git /home/azureuser/LiveCodeBench
    cd /home/azureuser/LiveCodeBench && pip install -e .

    # Run evaluation:
    python -m eval.livecodebench_run --model Qwen/Qwen3-4B-Instruct-2507
    python -m eval.livecodebench_run --model /path/to/finetuned --output-file lcb_finetuned.json

Requirements:
    - LiveCodeBench repo cloned and installed (pip install -e .)
    - vllm (used internally by LiveCodeBench for open models)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LCB_DEFAULT_PATH = "/home/azureuser/LiveCodeBench"


def run_livecodebench(model_path: str, lcb_path: str, tp_size: int = 1, release_version: str = "release_latest") -> dict:
    """Run LiveCodeBench evaluation via its CLI."""
    if not Path(lcb_path).exists():
        print(f"Error: LiveCodeBench not found at {lcb_path}")
        print(f"Clone it: git clone https://github.com/LiveCodeBench/LiveCodeBench.git {lcb_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", model_path,
        "--scenario", "codegeneration",
        "--evaluate",
        "--release_version", release_version,
        "--tensor_parallel_size", str(tp_size),
    ]

    print(f"Running LiveCodeBench...")
    print(f"  Model: {model_path}")
    print(f"  Release: {release_version}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=lcb_path,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_ALLOW_CODE_EVAL": "1"},
    )

    if result.returncode != 0:
        print(f"LiveCodeBench failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout[-2000:]}")
        print(f"STDERR: {result.stderr[-2000:]}")
        return {"error": result.stderr[-2000:]}

    print(result.stdout[-3000:])
    return {"stdout": result.stdout, "returncode": result.returncode}


def save_results(model_path: str, lcb_output: dict, output_dir: str, output_file: str | None) -> str:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        filename = output_file if output_file.endswith(".json") else f"{output_file}.json"
    else:
        model_name = Path(model_path).name.replace("/", "_")
        filename = f"livecodebench_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "benchmark": "livecodebench",
        "model": model_path,
        "timestamp": timestamp,
        "output": lcb_output,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run LiveCodeBench evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--lcb-path", default=LCB_DEFAULT_PATH, help="Path to cloned LiveCodeBench repo")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--release-version", default="release_latest", help="Dataset version (release_v1, release_v2, release_latest)")
    parser.add_argument("--output-dir", default="eval/results", help="Output directory")
    parser.add_argument("--output-file", default=None, help="Output filename")
    args = parser.parse_args()

    lcb_output = run_livecodebench(args.model, args.lcb_path, args.tp_size, args.release_version)
    save_results(args.model, lcb_output, args.output_dir, args.output_file)


if __name__ == "__main__":
    main()
