"""
Merge LoRA adapter weights with base model and save full merged models.

For each subfolder in replay_lora/ that has a saved_model/ directory,
merges the adapter with Qwen3-4B-Instruct-2507 and saves the full model
to saved_full_model/ beside the saved_model/ directory.

Usage:
    python merge_lora.py
"""

import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ROOT = "/home/azureuser/nvme_0/replay_lora"


def merge_and_save(adapter_path: str, output_path: str):
    """Load base model, merge adapter, and save full model."""
    print(f"  Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"  Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Merging and unloading...")
    model = model.merge_and_unload()

    print(f"  Saving to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Done!\n")


def main():
    lora_root = Path(LORA_ROOT)
    subdirs = sorted([d for d in lora_root.iterdir() if d.is_dir()])

    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA root: {LORA_ROOT}")
    print(f"Found {len(subdirs)} subfolders\n")

    for i, subdir in enumerate(subdirs, 1):
        adapter_path = subdir / "saved_model"
        output_path = subdir / "saved_full_model"

        if not adapter_path.exists():
            print(f"[{i}/{len(subdirs)}] Skipping {subdir.name} (no saved_model/)")
            continue

        if output_path.exists():
            print(f"[{i}/{len(subdirs)}] Skipping {subdir.name} (saved_full_model/ already exists)")
            continue

        print(f"[{i}/{len(subdirs)}] Merging: {subdir.name}")
        merge_and_save(str(adapter_path), str(output_path))

    print("All done!")


if __name__ == "__main__":
    main()
