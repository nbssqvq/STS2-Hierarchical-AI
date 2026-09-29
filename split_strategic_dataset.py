from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict


def _screen(example: Dict[str, Any]) -> str:
    try:
        user_message = next(message for message in example["messages"] if message["role"] == "user")
        return str(json.loads(user_message["content"]).get("screen", "unknown"))
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"


def split_by_run(input_path: str | Path, output_dir: str | Path, seed: int = 42,
                 train_ratio: float = 0.8, validation_ratio: float = 0.1) -> Dict[str, Any]:
    source = Path(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    by_run: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        run_id = str(example.get("metadata", {}).get("run_id", ""))
        if not run_id:
            raise ValueError("Every SFT example must contain metadata.run_id")
        by_run[run_id].append(example)

    run_ids = sorted(by_run)
    random.Random(seed).shuffle(run_ids)
    train_end = round(len(run_ids) * train_ratio)
    validation_end = train_end + round(len(run_ids) * validation_ratio)
    assignments = {
        "train": set(run_ids[:train_end]),
        "validation": set(run_ids[train_end:validation_end]),
        "test": set(run_ids[validation_end:]),
    }
    if ((assignments["train"] & assignments["validation"])
            or (assignments["train"] & assignments["test"])
            or (assignments["validation"] & assignments["test"])):
        raise RuntimeError("Run-level dataset split overlap detected")

    summary: Dict[str, Any] = {"input": str(source), "seed": seed, "splits": {}}
    for name, assigned_runs in assignments.items():
        selected = [example for run_id in run_ids if run_id in assigned_runs
                    for example in by_run[run_id]]
        destination = output / f"{name}.jsonl"
        destination.write_text("".join(
            json.dumps(example, ensure_ascii=False) + "\n" for example in selected
        ), encoding="utf-8")
        summary["splits"][name] = {
            "runs": len(assigned_runs),
            "examples": len(selected),
            "screens": dict(Counter(_screen(example) for example in selected)),
        }
    summary["total_runs"] = len(run_ids)
    summary["total_examples"] = len(examples)
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split strategic SFT examples without run leakage.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(split_by_run(args.input, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
