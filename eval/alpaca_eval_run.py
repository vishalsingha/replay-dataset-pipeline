"""
AlpacaEval 2.0: instruction-following quality measured by LLM-as-judge win-rate.

Workflow:
  1. Generate model responses to 805 AlpacaEval instructions using vLLM
  2. Judge responses vs reference (GPT-4 Preview) using GPT-4 Turbo
  3. Report length-controlled win rate

Usage:
    python -m eval.alpaca_eval_run --model Qwen/Qwen3-4B-Instruct-2507
    python -m eval.alpaca_eval_run --model /path/to/finetuned --output-file alpaca_finetuned.json

Requirements:
    pip install alpaca-eval vllm
    export OPENAI_API_KEY=<your-key>
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def generate_outputs(model_path: str, tp_size: int = 1) -> list[dict]:
    """Generate model responses to AlpacaEval instructions using vLLM."""
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("Loading AlpacaEval dataset...")
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)

    prompts = []
    instructions = []
    for example in ds:
        instruction = example["instruction"]
        instructions.append(instruction)
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"Generating responses for {len(prompts)} instructions...")
    outputs = llm.generate(prompts, sampling)

    model_outputs = []
    for instruction, output in zip(instructions, outputs):
        response = output.outputs[0].text.strip()
        model_outputs.append({
            "instruction": instruction,
            "output": response,
            "generator": Path(model_path).name,
        })

    print(f"Generated {len(model_outputs)} responses")
    return model_outputs


def run_alpaca_eval(model_outputs: list[dict], output_dir: str) -> dict:
    """Run AlpacaEval judgment on generated outputs."""
    try:
        import alpaca_eval
    except ImportError:
        print("Error: alpaca-eval not installed. Run: pip install alpaca-eval")
        sys.exit(1)

    from alpaca_eval import evaluate

    outputs_path = os.path.join(output_dir, "_alpaca_eval_outputs.json")
    with open(outputs_path, "w") as f:
        json.dump(model_outputs, f, indent=2)

    print("Running AlpacaEval judgment (this calls GPT-4 Turbo)...")
    results = evaluate(
        model_outputs=model_outputs,
        output_path=output_dir,
    )
    return results


def save_results(model_path: str, model_outputs: list[dict], eval_results: dict | None, output_dir: str, output_file: str | None) -> str:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        filename = output_file if output_file.endswith(".json") else f"{output_file}.json"
    else:
        model_name = Path(model_path).name.replace("/", "_")
        filename = f"alpaca_eval_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "benchmark": "alpaca_eval_2",
        "model": model_path,
        "timestamp": timestamp,
        "num_instructions": len(model_outputs),
        "results": eval_results if eval_results else "generate_only",
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run AlpacaEval 2.0 evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--output-dir", default="eval/results", help="Output directory")
    parser.add_argument("--output-file", default=None, help="Output filename")
    parser.add_argument("--generate-only", action="store_true", help="Only generate outputs, skip judging")
    args = parser.parse_args()

    if not args.generate_only and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for AlpacaEval judging.")
        print("Set it or use --generate-only to just generate outputs.")
        sys.exit(1)

    model_outputs = generate_outputs(args.model, args.tp_size)

    if args.generate_only:
        save_results(args.model, model_outputs, None, args.output_dir, args.output_file)
        print("Generation complete. Re-run without --generate-only to judge.")
        return

    eval_results = run_alpaca_eval(model_outputs, args.output_dir)
    save_results(args.model, model_outputs, eval_results, args.output_dir, args.output_file)


if __name__ == "__main__":
    main()
