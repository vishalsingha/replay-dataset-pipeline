"""
Stage 2: Mix replay with domain data

Combines single-turn replay + multi-turn replay + domain SFT data at a
configurable ratio (default: 17% domain / 83% replay), shuffles,
and optionally creates a validation split.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.utils import atomic_write_jsonl, load_jsonl


def mix_datasets(config_path: str) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    mcfg = cfg["mixing"]
    domain_fraction = mcfg["domain_fraction"]
    domain_path = mcfg["domain_data_path"]
    val_fraction = mcfg.get("val_fraction", 0.0)
    seed = mcfg.get("seed", 42)

    replay_path = Path(cfg["paths"]["replay"])
    public_replay_path = Path(cfg["paths"].get("public_replay", ""))
    multiturn_path = Path(cfg["paths"]["multiturn"])
    train_path = Path(cfg["paths"]["train"])
    val_path = Path(cfg["paths"]["val"])
    train_path.parent.mkdir(parents=True, exist_ok=True)

    if domain_path.startswith("PLACEHOLDER"):
        raise ValueError(
            "Set mixing.domain_data_path in config.yaml to your domain dataset .jsonl path"
        )

    if domain_fraction <= 0 or domain_fraction >= 1:
        raise ValueError(f"domain_fraction must be between 0 and 1 (exclusive), got {domain_fraction}")

    # Load all replay sources
    log.info("Loading replay sources...")

    log.info(f"  Single-turn replay: {replay_path}")
    replay_single = load_jsonl(replay_path, required_key="messages")
    log.info(f"    {len(replay_single)} examples")

    log.info(f"  Public replay: {public_replay_path}")
    replay_public = load_jsonl(public_replay_path, required_key="messages")
    log.info(f"    {len(replay_public)} examples")

    log.info(f"  Multi-turn replay: {multiturn_path}")
    replay_multi = load_jsonl(multiturn_path, required_key="messages")
    log.info(f"    {len(replay_multi)} examples")

    replay_data = replay_single + replay_public + replay_multi
    log.info(f"  Total replay: {len(replay_data)} examples")

    log.info(f"Loading domain data from {domain_path}...")
    domain_data = load_jsonl(domain_path, required_key="messages")
    log.info(f"  {len(domain_data)} domain examples")

    if not domain_data:
        log.error("Domain dataset is empty or has no valid entries with 'messages' key. "
                  "Cannot compute mixing ratio. Aborting.")
        return

    total_target = int(len(domain_data) / domain_fraction)
    replay_needed = total_target - len(domain_data)

    if replay_needed > len(replay_data):
        log.warning(
            f"Need {replay_needed} replay examples for {domain_fraction:.0%} domain ratio, "
            f"but only have {len(replay_data)}. Using all replay data."
        )
        replay_sample = replay_data
    elif replay_needed < len(replay_data):
        log.info(f"Sampling {replay_needed} from {len(replay_data)} replay examples...")
        random.seed(seed)
        replay_sample = random.sample(replay_data, replay_needed)
    else:
        replay_sample = replay_data

    combined = replay_sample + domain_data
    random.seed(seed)
    random.shuffle(combined)

    actual_domain_frac = len(domain_data) / len(combined)
    n_single = sum(1 for r in replay_sample if sum(1 for m in r["messages"] if m["role"] == "user") == 1)
    n_multi = len(replay_sample) - n_single
    log.info(
        f"Combined: {len(combined)} examples "
        f"(domain={len(domain_data)} [{actual_domain_frac:.1%}], "
        f"replay={len(replay_sample)} [{1 - actual_domain_frac:.1%}] "
        f"— {n_single} single-turn + {n_multi} multi-turn)"
    )

    if val_fraction > 0:
        val_size = int(len(combined) * val_fraction)
        val_data = combined[:val_size]
        train_data = combined[val_size:]
        log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

        log.info(f"Writing validation set to {val_path} (atomic)")
        atomic_write_jsonl(val_path, val_data)
    else:
        train_data = combined

    log.info(f"Writing training set to {train_path} (atomic)")
    atomic_write_jsonl(train_path, train_data)

    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Mix replay + domain data for training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    mix_datasets(args.config)


if __name__ == "__main__":
    main()
