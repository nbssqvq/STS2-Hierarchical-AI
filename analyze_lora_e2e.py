from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PATTERNS = {
    "llm_decisions": r"\[LLM_DECISION\]",
    "llm_fallbacks": r"\[LLM_FALLBACK\]",
    "llm_retries": r"\[LLM_RETRY\]",
    "llm_invalid": r"\[LLM_INVALID\]",
    "llm_stale": r"\[LLM_STALE\]",
    "strategy_guards": r"\[STRATEGY_GUARD\]",
    "shop_reserve_guards": r"\[SHOP_RESERVE_GUARD\]",
    "auto_retries": r"\[AUTO_RETRY\]",
    "action_rejected": r"ACTION_REJECTED",
    "episode_failed": r"\[EPISODE_FAILED\]",
    "auto_stalls": r"auto_step_count >",
}


def describe(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "median": round(statistics.median(values), 3) if values else None,
        "min": min(values, default=None),
        "max": max(values, default=None),
    }


def analyze(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = (raw.decode("utf-16-le", errors="replace") if raw[:512].count(b"\x00") > 32
            else raw.decode("utf-8-sig", errors="replace"))
    result_line = next(
        (line for line in reversed(text.splitlines()) if line.startswith('[{"final_screen"')),
        None,
    )
    if result_line is None:
        raise RuntimeError("Final episode result array was not found")
    episodes = json.loads(result_line)
    floors = [int(row["max_floor_reached"]) for row in episodes]
    rewards = [float(row["total_reward"]) for row in episodes]
    steps = [int(row["steps"]) for row in episodes]
    auto_steps = [int(row["auto_steps"]) for row in episodes]
    latencies = [float(value) for value in re.findall(r"latency=([0-9.]+)s", text)]
    screens = Counter(re.findall(r"\[LLM_DECISION\].*?screen=([^ ]+)", text))
    terminal = Counter(str(row["final_screen"]) for row in episodes)
    return {
        "episodes": len(episodes),
        "floors": describe([float(value) for value in floors]),
        "floor_distribution": dict(sorted(Counter(floors).items())),
        "rewards": describe(rewards),
        "combat_steps": describe([float(value) for value in steps]),
        "auto_steps": describe([float(value) for value in auto_steps]),
        "latency_seconds": describe(latencies),
        "decision_screens": dict(screens.most_common()),
        "final_screens": dict(terminal),
        "counts": {name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()},
        "boss_reach_rate": round(sum(floor_value >= 17 for floor_value in floors) / len(floors), 4),
        "act2_reach_rate": round(sum(floor_value >= 18 for floor_value in floors) / len(floors), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="logs/lora_e2e_20games.log")
    parser.add_argument("--output", default="logs/lora_e2e_20games.summary.json")
    args = parser.parse_args()
    summary = analyze(Path(args.log))
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
