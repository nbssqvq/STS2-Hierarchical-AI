from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sts2_rl.strategy.ollama_strategic import SYSTEM_PROMPT, strategy_prompt
from sts2_rl.strategy.strategic_routing import StrategicPolicyGuard


SAFE_ROUTES = {"rest", "rest_site", "campfire", "fire", "shop", "event", "unknown", "monster"}
DANGEROUS_ROUTES = {"elite", "boss"}
REVIEW_ONLY_REASONS = {
    "basic_upgrade_with_nonbasic_alternative_needs_review",
    "event_cost_would_be_lethal",
    "event_hp_after_payment_low_needs_review",
    "high_hp_rest_with_growth_option_needs_review",
    "skipped_priority_card_removal",
    "elite_without_health_margin",
    "low_hp_chose_danger_with_safe_route_available",
    "low_hp_failed_to_rest",
    "large_deck_added_card_needs_review",
}


def _candidate(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    observation = record.get("observation", {})
    action = record.get("action", {})
    candidates = observation.get("candidates", {})
    name = action.get("action")
    index = action.get("index", action.get("card_index"))
    if not isinstance(index, int):
        return None
    if name == "choose_map_node":
        items = candidates.get("map_nodes", [])
    elif name == "shop_purchase":
        items = candidates.get("items", [])
    elif name in {"select_card_reward", "select_card", "combat_select_card"}:
        items = candidates.get("cards", [])
    else:
        items = candidates.get("options", [])
    for position, item in enumerate(items if isinstance(items, list) else []):
        if isinstance(item, dict) and int(item.get("index", position)) == index:
            return item
    return None


def _text(item: Optional[Dict[str, Any]]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(str(item.get(key, "")) for key in (
        "id", "title", "name", "type", "category", "description", "card_name",
    )).lower()


def _hp_ratio(observation: Dict[str, Any]) -> float:
    player = observation.get("player", {})
    try:
        return float(player.get("hp", 0) or 0) / max(1.0, float(player.get("max_hp", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _immediate_score(record: Dict[str, Any]) -> tuple[float, list[str]]:
    source = str(record.get("source", ""))
    if not record.get("accepted"):
        return -20.0, ["not_an_accepted_ollama_action"]
    if source in {"shop_reserve_guard", "strategy_guard"}:
        score = 4.0
        reasons: list[str] = ["accepted_rule_corrected_action"]
    elif source == "ollama":
        score = 0.0
        reasons = ["accepted_ollama_action"]
    else:
        return -20.0, ["not_an_accepted_ollama_action"]
    observation = record.get("observation", {})
    action = record.get("action", {})
    screen = str(record.get("screen", ""))
    selected = _candidate(record)
    selected_text = _text(selected)

    if screen in {"map", "map_screen"}:
        route = str((selected or {}).get("type", "")).lower()
        hp_ratio = _hp_ratio(observation)
        options = observation.get("candidates", {}).get("map_nodes", [])
        option_types = {str(item.get("type", "")).lower() for item in options if isinstance(item, dict)}
        if hp_ratio < 0.40 and route in DANGEROUS_ROUTES and option_types & SAFE_ROUTES:
            score -= 10.0
            reasons.append("low_hp_chose_danger_with_safe_route_available")
        elif hp_ratio < 0.40 and route in {"rest", "rest_site", "campfire", "fire"}:
            score += 4.0
            reasons.append("low_hp_chose_rest")
        elif route == "elite" and hp_ratio < 0.70:
            score -= 3.0
            reasons.append("elite_without_health_margin")
        elif route in SAFE_ROUTES:
            score += 1.0
            reasons.append("reasonable_route")

    elif screen in {"shop", "shop_screen", "fake_merchant"}:
        items = observation.get("candidates", {}).get("items", [])
        affordable_removals = [item for item in items if isinstance(item, dict)
                               and "card_removal" in _text(item)
                               and item.get("can_afford") is True
                               and item.get("is_stocked") is not False]
        if "card_removal" in selected_text:
            score += 4.0
            reasons.append("bought_card_removal")
        elif affordable_removals and action.get("action") == "shop_purchase":
            score -= 3.0
            reasons.append("skipped_priority_card_removal")

    elif screen in {"rest", "rest_site", "campfire", "fire"}:
        hp_ratio = _hp_ratio(observation)
        is_rest = any(term in selected_text for term in ("rest", "heal", "休息", "治疗", "恢复"))
        is_upgrade = any(term in selected_text for term in ("upgrade", "smith", "升级", "锻造"))
        options = observation.get("candidates", {}).get("options", [])
        growth_available = any(
            isinstance(option, dict) and option != selected
            and option.get("is_locked") is not True and option.get("enabled") is not False
            and option.get("can_select") is not False
            and any(term in _text(option) for term in ("upgrade", "smith", "升级", "锻造"))
            for option in (options if isinstance(options, list) else [])
        )
        if hp_ratio < 0.40:
            score += 5.0 if is_rest else -10.0
            reasons.append("low_hp_rest" if is_rest else "low_hp_failed_to_rest")
        elif hp_ratio > 0.75 and is_rest and growth_available:
            reasons.append("high_hp_rest_with_growth_option_needs_review")
        elif hp_ratio > 0.75 and is_upgrade:
            score += 2.0
            reasons.append("healthy_upgrade")

    elif screen == "card_reward":
        deck = observation.get("player", {}).get("deck", [])
        deck_size = len(deck) if isinstance(deck, list) else 0
        if action.get("action") == "skip_card_reward":
            score += 2.0 if deck_size > 20 else 0.5
            reasons.append("skipped_card_reward")
        elif deck_size > 20:
            score -= 2.0
            reasons.append("large_deck_added_card_needs_review")
        else:
            score += 0.5
            reasons.append("card_added_to_small_deck")

    elif screen in {"event", "event_screen"}:
        player = observation.get("player", {})
        hp = float(player.get("hp", 0) or 0)
        max_hp = float(player.get("max_hp", 0) or 0)
        cost = StrategicPolicyGuard._event_hp_cost(selected or {})
        remaining = hp - cost
        if cost > 0 and remaining <= 0:
            score -= 20.0
            reasons.append("event_cost_would_be_lethal")
        elif cost > 0 and (remaining < 30 or (max_hp > 0 and remaining / max_hp < 0.5)):
            score -= 8.0
            reasons.append("event_hp_after_payment_low_needs_review")
        else:
            score += 0.5
            reasons.append("nonlethal_event_choice")

    elif screen in {"hand_select", "card_select", "card_grid_selection", "choose_a_card"}:
        screen_type = str(observation.get("screen_meta", {}).get("screen_type") or "").lower()
        prompt = str(observation.get("prompt") or "").lower()
        is_upgrade = screen_type == "upgrade" or "升级" in prompt or "upgrade" in prompt
        basic_ids = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}
        alternatives = observation.get("candidates", {}).get("cards", [])
        nonbasic_available = any(
            isinstance(card, dict) and card.get("id") not in basic_ids
            and card.get("upgraded") is not True and card.get("can_select") is not False
            for card in alternatives
        )
        if is_upgrade and (selected or {}).get("id") in basic_ids and nonbasic_available:
            score -= 20.0
            reasons.append("basic_upgrade_with_nonbasic_alternative_needs_review")
        else:
            score += 0.5
            reasons.append("valid_card_selection")

    return score, reasons


def score_run(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    end = next((row for row in reversed(records) if row.get("record_type") == "episode_end"), {})
    max_floor = int(end.get("max_floor", 0) or 0)
    victory = bool(end.get("victory"))
    scored = []
    for row in records:
        if row.get("record_type") != "strategic_decision":
            continue
        immediate, reasons = _immediate_score(row)
        floor = int(row.get("floor", 0) or 0)
        floors_survived = max(0, max_floor - floor)
        delayed = min(3.0, floors_survived * 0.25)
        if max_floor >= 17 and floor < 17:
            delayed += 2.0
            reasons.append("run_reached_boss")
        if victory:
            delayed += 5.0
            reasons.append("run_victory")
        if floors_survived == 0 and not victory:
            delayed -= 2.0
            reasons.append("run_ended_on_decision_floor")
        item = dict(row)
        item["score"] = round(immediate + delayed, 3)
        item["immediate_score"] = round(immediate, 3)
        item["delayed_score"] = round(delayed, 3)
        item["floors_survived"] = floors_survived
        item["score_reasons"] = reasons
        scored.append(item)
    return scored


def load_runs(paths: Iterable[Path]) -> Iterable[list[Dict[str, Any]]]:
    for path in paths:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if rows:
            yield rows


def build_dataset(input_dir: str | Path, output_dir: str | Path,
                  sft_threshold: float = 2.0,
                  run_paths: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scored_path = output_path / "scored_decisions.jsonl"
    sft_path = output_path / "sft.jsonl"
    review_path = output_path / "needs_review.jsonl"
    counts: Counter[str] = Counter()
    review_flags: Counter[str] = Counter()
    with scored_path.open("w", encoding="utf-8") as scored_stream, \
            sft_path.open("w", encoding="utf-8") as sft_stream, \
            review_path.open("w", encoding="utf-8") as review_stream:
        for run in load_runs(run_paths if run_paths is not None
                             else sorted(input_path.glob("*.jsonl"))):
            if not any(row.get("record_type") == "episode_end" for row in run):
                counts["incomplete_runs_skipped"] += 1
                continue
            counts["runs"] += 1
            for row in score_run(run):
                counts["decisions"] += 1
                scored_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                is_candidate = (row.get("sft_candidate", row.get("llm_eligible_positive", False))
                                or row.get("source") in {"shop_reserve_guard", "strategy_guard"})
                if (is_candidate and row["score"] >= sft_threshold
                        and not REVIEW_ONLY_REASONS.intersection(row["score_reasons"])):
                    observation = row["observation"]
                    example = {"messages": [
                        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{strategy_prompt(observation)}"},
                        {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
                        {"role": "assistant", "content": json.dumps(
                            {"status": "act", "action": row["action"]}, ensure_ascii=False)},
                    ], "metadata": {"run_id": row.get("run_id"), "sequence": row.get("sequence"),
                                     "score": row["score"], "reasons": row["score_reasons"]}}
                    sft_stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts["sft"] += 1
                else:
                    review_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["review"] += 1
                    review_flags.update(REVIEW_ONLY_REASONS.intersection(row["score_reasons"]))
    summary = {"input": str(input_path), "output": str(output_path),
               "sft_threshold": sft_threshold, **counts,
               "review_flags": dict(sorted(review_flags.items()))}
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score strategic traces and export Qwen SFT examples.")
    parser.add_argument("--input", default="logs/strategic_trajectories")
    parser.add_argument("--output", default="logs/strategic_dataset")
    parser.add_argument("--sft-threshold", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.input, args.output, args.sft_threshold),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
