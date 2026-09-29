"""Mix verified upgrade corrections with non-upgrade replay examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_upgrade(example: dict[str, Any]) -> bool:
    messages = example.get("messages", [])
    if len(messages) < 2:
        return False
    try:
        observation = json.loads(messages[1]["content"])
    except (KeyError, TypeError, ValueError):
        return False
    if observation.get("screen") not in {"card_select", "card_grid_selection"}:
        return False
    meta = observation.get("screen_meta", {})
    prompt = str(observation.get("prompt") or "").lower()
    return meta.get("screen_type") == "upgrade" or "升级" in prompt or "upgrade" in prompt


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def build(corrections: Path, old_train: Path, old_validation: Path,
          output_dir: Path, seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)
    curated = _read(corrections)
    rng.shuffle(curated)
    if len(curated) < 5:
        raise ValueError("At least five verified corrections are required before fine-tuning")
    validation_count = max(2, round(len(curated) * 0.2))
    upgrade_train = curated[validation_count:]
    upgrade_validation = curated[:validation_count]
    replay_train = [row for row in _read(old_train) if not _is_upgrade(row)]
    replay_validation = [row for row in _read(old_validation) if not _is_upgrade(row)]
    rng.shuffle(replay_train)
    rng.shuffle(replay_validation)
    # Repeat only the narrow verified corrections; retain a larger sample of
    # other decisions to reduce drift in map, shop and event behavior.
    train = upgrade_train * 8 + replay_train[:200]
    validation = upgrade_validation + replay_validation[:50]
    rng.shuffle(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "train.jsonl", train)
    _write(output_dir / "validation.jsonl", validation)
    _write(output_dir / "upgrade_holdout.jsonl", upgrade_validation)
    summary = {"verified_corrections": len(curated), "upgrade_train_unique": len(upgrade_train),
               "upgrade_train_repeated": len(upgrade_train) * 8,
               "upgrade_holdout": len(upgrade_validation),
               "replay_train": min(200, len(replay_train)),
               "replay_validation": min(50, len(replay_validation)),
               "train_total": len(train), "validation_total": len(validation)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare upgrade correction LoRA data.")
    parser.add_argument("--corrections", type=Path, default=Path("logs/upgrade_corrections/sft.jsonl"))
    parser.add_argument("--old-train", type=Path,
                        default=Path("logs/strategic_dataset_qwen_300/split/train.jsonl"))
    parser.add_argument("--old-validation", type=Path,
                        default=Path("logs/strategic_dataset_qwen_300/split/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/upgrade_finetune"))
    args = parser.parse_args()
    print(json.dumps(build(args.corrections, args.old_train, args.old_validation, args.output_dir)))
