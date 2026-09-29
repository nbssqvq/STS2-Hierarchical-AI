"""Read-only summary of completed strategic trajectory runs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median

from strategic_dataset import REVIEW_ONLY_REASONS, build_dataset, score_run
from strategic_routing import StrategicPolicyGuard


def selected_option(record: dict) -> dict:
    action = record.get("action") or {}
    index = action.get("index", action.get("card_index"))
    candidates = record.get("observation", {}).get("candidates", {})
    action_name = action.get("action")
    if action_name == "choose_map_node":
        options = candidates.get("map_nodes", [])
    elif action_name == "shop_purchase":
        options = candidates.get("items", [])
    elif action_name in {"select_card_reward", "select_card", "combat_select_card"}:
        options = candidates.get("cards", [])
    else:
        options = candidates.get("options", [])
    for position, option in enumerate(options if isinstance(options, list) else []):
        if isinstance(option, dict) and option.get("index", position) == index:
            return option
    return {}


def completed_run_paths(root: Path, after: datetime, count: int) -> list[Path]:
    files = sorted((path for path in root.glob("*.jsonl")
                    if datetime.fromtimestamp(path.stat().st_mtime) >= after),
                   key=lambda path: path.stat().st_mtime)
    paths = []
    for path in files:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if any(row.get("record_type") == "episode_end" for row in rows):
            paths.append(path)
        if len(paths) >= count:
            break
    return paths


def summarize(root: Path, after: datetime, count: int) -> dict:
    runs = []
    for path in completed_run_paths(root, after, count):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        ends = [row for row in rows if row.get("record_type") == "episode_end"]
        runs.append((path, rows, ends[-1]))
    decisions = [row for _, rows, _ in runs for row in rows
                 if row.get("record_type") == "strategic_decision"]
    floors = [end["max_floor"] for _, _, end in runs]
    risky_events = []
    high_hp_rest = []
    low_hp_nonrest = []
    low_hp_elite = []
    shop_actions = Counter()
    high_hp_rest_with_alternative = 0
    high_hp_rest_at_full_hp = 0
    scored = [row for _, rows, _ in runs for row in score_run(rows)]
    scored_by_key = {(row.get("run_id"), row.get("sequence")): row for row in scored}
    risky_event_sft_eligible = 0
    high_hp_rest_sft_eligible = 0
    for row in decisions:
        screen = row.get("screen")
        observation = row.get("observation") or {}
        player = observation.get("player") or {}
        hp = float(player.get("hp") or 0)
        max_hp = max(float(player.get("max_hp") or 0), 1)
        choice = selected_option(row)
        scored_row = scored_by_key.get((row.get("run_id"), row.get("sequence")), {})
        sft_eligible = (row.get("sft_candidate") and scored_row.get("score", -100) >= 2.0
                        and REVIEW_ONLY_REASONS.isdisjoint(scored_row.get("score_reasons", [])))
        if screen in {"event", "event_screen"}:
            cost = StrategicPolicyGuard._event_hp_cost(choice)
            if cost and (hp - cost <= 0 or (hp - cost) / max_hp < 0.5):
                risky_event_sft_eligible += bool(sft_eligible)
                risky_events.append({"run_id": row.get("run_id"), "floor": row.get("floor"),
                                     "hp": hp, "max_hp": max_hp, "cost": cost,
                                     "choice": choice.get("description"),
                                     "source": row.get("source")})
        if screen in {"rest", "rest_site", "campfire", "fire"}:
            label = str(choice.get("title") or choice.get("name") or choice.get("description") or "")
            rested = any(term in label.lower() for term in ("rest", "heal", "休息", "恢复"))
            example = {"floor": row.get("floor"), "hp": hp, "max_hp": max_hp,
                       "choice": label, "reason": row.get("reason")}
            if hp / max_hp > 0.75 and rested:
                high_hp_rest_sft_eligible += bool(sft_eligible)
                high_hp_rest.append(example)
                if len(observation.get("candidates", {}).get("options", [])) > 1:
                    high_hp_rest_with_alternative += 1
                if hp >= max_hp:
                    high_hp_rest_at_full_hp += 1
            if hp / max_hp < 0.5 and not rested:
                low_hp_nonrest.append(example)
        if screen in {"map", "map_screen"} and hp / max_hp < 0.7:
            if str(choice.get("type") or "").lower() == "elite":
                low_hp_elite.append({"floor": row.get("floor"), "hp": hp, "max_hp": max_hp,
                                     "options": observation.get("candidates", {}).get("map_nodes", [])})
        if screen in {"shop", "shop_screen", "fake_merchant"}:
            shop_actions[str(choice.get("category") or row.get("action", {}).get("action"))] += 1
    return {
        "runs": len(runs), "first_run": runs[0][0].name if runs else None,
        "last_run": runs[-1][0].name if runs else None,
        "mean_floor": round(mean(floors), 2) if floors else None,
        "median_floor": median(floors) if floors else None,
        "max_floor": max(floors) if floors else None,
        "floor_buckets": dict(Counter("1-5" if floor <= 5 else "6-10" if floor <= 10
                                      else "11-16" if floor <= 16 else "17+" for floor in floors)),
        "victories": sum(bool(end.get("victory")) for _, _, end in runs),
        "mean_reward": round(mean(end["episode_reward"] for _, _, end in runs), 2) if runs else None,
        "decisions": len(decisions), "sources": dict(Counter(row.get("source") for row in decisions)),
        "screens": dict(Counter(row.get("screen") for row in decisions)),
        "accepted": sum(bool(row.get("accepted")) for row in decisions),
        "shop_actions": dict(shop_actions),
        "sft_eligible": sum(row.get("sft_candidate") and row.get("score", -100) >= 2.0
                            and REVIEW_ONLY_REASONS.isdisjoint(row.get("score_reasons", []))
                            for row in scored),
        "risky_event_sft_eligible": risky_event_sft_eligible,
        "high_hp_rest_sft_eligible": high_hp_rest_sft_eligible,
        "high_hp_rest_with_alternative": high_hp_rest_with_alternative,
        "high_hp_rest_at_full_hp": high_hp_rest_at_full_hp,
        "risky_events": risky_events, "high_hp_rest": high_hp_rest,
        "low_hp_nonrest": low_hp_nonrest, "low_hp_elite": low_hp_elite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/strategic_trajectories")
    parser.add_argument("--after", default="2026-09-24T14:20:00")
    parser.add_argument("--runs", type=int, default=67)
    parser.add_argument("--export", help="Export only these completed runs with the current SFT filter.")
    args = parser.parse_args()
    root = Path(args.input)
    after = datetime.fromisoformat(args.after)
    summary = summarize(root, after, args.runs)
    if args.export:
        summary["dataset"] = build_dataset(
            root, args.export, run_paths=completed_run_paths(root, after, args.runs))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
