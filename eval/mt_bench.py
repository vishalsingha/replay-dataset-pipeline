"""
MT-Bench evaluation: multi-turn conversation quality scored by LLM-as-judge.

Workflow:
  1. Generate model answers to 80 multi-turn questions using vLLM
  2. Judge answers using GPT-4 (requires OPENAI_API_KEY)
  3. Report average scores (1-10) per category and overall

Usage:
    python -m eval.mt_bench --model Qwen/Qwen3-4B-Instruct-2507
    python -m eval.mt_bench --model /path/to/finetuned --output-file mt_bench_finetuned.json

Requirements:
    pip install fschat[model_worker,llm_judge] vllm
    export OPENAI_API_KEY=<your-key>
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MT_BENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]


def load_questions() -> list[dict]:
    """Load MT-Bench questions (80 multi-turn questions)."""
    import urllib.request
    cache_path = Path("eval/data/mt_bench_questions.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print("Downloading MT-Bench questions...")
        urllib.request.urlretrieve(MT_BENCH_QUESTIONS_URL, cache_path)

    questions = []
    with open(cache_path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def generate_answers(model_path: str, questions: list[dict], tp_size: int = 1) -> list[dict]:
    """Generate model answers using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)

    answers = []
    for q in questions:
        turns = q["turns"]
        conversation = []
        model_responses = []

        for turn_idx, turn_text in enumerate(turns):
            conversation.append({"role": "user", "content": turn_text})
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            output = llm.generate([prompt], sampling)[0]
            response = output.outputs[0].text.strip()
            model_responses.append(response)
            conversation.append({"role": "assistant", "content": response})

        answers.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "turns": turns,
            "model_responses": model_responses,
        })
        print(f"  [{len(answers)}/{len(questions)}] Q{q['question_id']} ({q['category']})")

    return answers


def judge_answers(answers: list[dict], judge_model: str = "gpt-4o") -> list[dict]:
    """Score answers using LLM-as-judge (supports OpenAI and Azure OpenAI)."""
    if os.environ.get("OPENAI_API_TYPE") == "azure" or os.environ.get("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
        judge_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", judge_model)
        print(f"  Using Azure OpenAI judge: {judge_model}")
    else:
        from openai import OpenAI
        client = OpenAI()
        print(f"  Using OpenAI judge: {judge_model}")

    results = []

    judge_prompt_template = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors including helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

    for i, ans in enumerate(answers):
        scores = []
        for turn_idx, (question, response) in enumerate(zip(ans["turns"], ans["model_responses"])):
            prompt = judge_prompt_template.format(question=question, answer=response)
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                judgment = resp.choices[0].message.content
                score = _extract_score(judgment)
                scores.append(score)
            except Exception as e:
                print(f"  Warning: judge failed for Q{ans['question_id']} turn {turn_idx+1}: {e}")
                scores.append(None)

        results.append({
            "question_id": ans["question_id"],
            "category": ans["category"],
            "scores": scores,
            "avg_score": sum(s for s in scores if s) / max(len([s for s in scores if s]), 1),
        })
        print(f"  [{i+1}/{len(answers)}] Q{ans['question_id']}: {scores}")

    return results


def _extract_score(judgment: str) -> float | None:
    """Extract numerical score from judge response."""
    import re
    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", judgment)
    if match:
        return float(match.group(1))
    match = re.search(r"Rating:\s*(\d+(?:\.\d+)?)", judgment)
    if match:
        return float(match.group(1))
    return None


def print_results(results: list[dict], model_path: str) -> dict:
    """Print and return summary statistics."""
    category_scores = {}
    for r in results:
        cat = r["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        if r["avg_score"]:
            category_scores[cat].append(r["avg_score"])

    print("\n" + "=" * 50)
    print(f"  MT-Bench Results: {Path(model_path).name}")
    print("=" * 50)
    print(f"  {'Category':<15} {'Avg Score':>10} {'Count':>8}")
    print("-" * 50)

    all_scores = []
    summary = {}
    for cat in CATEGORIES:
        scores = category_scores.get(cat, [])
        if scores:
            avg = sum(scores) / len(scores)
            all_scores.extend(scores)
            summary[cat] = avg
            print(f"  {cat:<15} {avg:>10.2f} {len(scores):>8}")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    summary["overall"] = overall
    print("-" * 50)
    print(f"  {'OVERALL':<15} {overall:>10.2f} {len(all_scores):>8}")
    print("=" * 50 + "\n")

    return summary


def save_results(model_path: str, answers: list[dict], results: list[dict], summary: dict, output_dir: str, output_file: str | None) -> str:
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        filename = output_file if output_file.endswith(".json") else f"{output_file}.json"
    else:
        model_name = Path(model_path).name.replace("/", "_")
        filename = f"mt_bench_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    output = {
        "benchmark": "mt_bench",
        "model": model_path,
        "timestamp": timestamp,
        "summary": summary,
        "results": results,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run MT-Bench evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model (default: gpt-4o)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--output-dir", default="eval/results", help="Output directory")
    parser.add_argument("--output-file", default=None, help="Output filename")
    parser.add_argument("--generate-only", action="store_true", help="Only generate answers, skip judging")
    args = parser.parse_args()

    if not args.generate_only and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for judging. Set it or use --generate-only.")
        sys.exit(1)

    questions = load_questions()
    print(f"Loaded {len(questions)} MT-Bench questions")

    print("\n--- Generating answers ---")
    answers = generate_answers(args.model, questions, args.tp_size)

    if args.generate_only:
        save_results(args.model, answers, [], {}, args.output_dir, args.output_file)
        print("Generation complete. Re-run without --generate-only to judge.")
        return

    print("\n--- Judging answers ---")
    results = judge_answers(answers, args.judge_model)
    summary = print_results(results, args.model)
    save_results(args.model, answers, results, summary, args.output_dir, args.output_file)


if __name__ == "__main__":
    main()
