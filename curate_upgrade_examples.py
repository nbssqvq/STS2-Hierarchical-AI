"""Create narrow, traceable LoRA corrections for upgrade selections.

Only unique, verified cost reductions and Armaments' single-to-all change are
used as automatic labels. Ambiguous choices are left in the review file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from ollama_strategic import SYSTEM_PROMPT, strategy_prompt
from upgrade_catalog import enrich_upgrade_cards


BASIC_IDS = {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}


def _upgrade_context(observation: dict[str, Any]) -> bool:
    if observation.get("screen") not in {"card_select", "card_grid_selection"}:
        return False
    screen_type = str(observation.get("screen_meta", {}).get("screen_type") or "").lower()
    prompt = str(observation.get("prompt") or "").lower()
    return screen_type == "upgrade" or "升级" in prompt or "upgrade" in prompt


def _clear_upgrade(card: dict[str, Any]) -> str | None:
    if card.get("id") in BASIC_IDS or card.get("upgraded") is True:
        return None
    preview = card.get("upgrade_preview")
    if not isinstance(preview, dict) or not preview.get("description"):
        return None
    try:
        before, after = int(card.get("cost")), int(preview.get("cost"))
        if after < before:
            return "verified_energy_cost_reduction"
    except (TypeError, ValueError):
        pass
    before_text = str(card.get("description") or "")
    after_text = str(preview.get("description") or "")
    if (card.get("id") == "ARMAMENTS" and "一张牌" in before_text
            and "所有牌" in after_text):
        return "verified_armaments_upgrades_all_hand_cards"
    return None


def curate(input_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[Any, ...]] = set()
    for path in sorted(input_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation", {})
            if record.get("record_type") != "strategic_decision" or not _upgrade_context(observation):
                continue
            counts["upgrade_decisions"] += 1
            cards = enrich_upgrade_cards(observation.get("candidates", {}).get("cards", []))
            options = [(card, reason) for card in cards if (reason := _clear_upgrade(card))]
            if len(options) != 1:
                counts["ambiguous"] += 1
                review.append({"run_id": record.get("run_id"), "floor": record.get("floor"),
                               "original_action": record.get("action"), "observation": observation,
                               "reason": "no_unique_verified_high_impact_upgrade"})
                continue
            selected, reason = options[0]
            if not isinstance(selected.get("index"), int):
                counts["invalid_index"] += 1
                continue
            signature = (observation.get("screen"), selected.get("id"), reason,
                         tuple((card.get("id"), card.get("upgraded")) for card in cards))
            if signature in seen:
                counts["duplicates"] += 1
                continue
            seen.add(signature)
            enriched = dict(observation)
            enriched["candidates"] = dict(observation.get("candidates", {}))
            enriched["candidates"]["cards"] = cards
            enriched["player"] = dict(observation.get("player", {}))
            enriched["player"]["deck"] = [
                {key: card.get(key) for key in ("id", "name", "upgraded")}
                for card in observation.get("player", {}).get("deck", [])
            ]
            action = {"action": "select_card", "index": selected["index"]}
            accepted.append({"messages": [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n{strategy_prompt(enriched)}"},
                {"role": "user", "content": json.dumps(enriched, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(
                    {"status": "act", "action": action}, ensure_ascii=False)},
            ], "metadata": {"run_id": record.get("run_id"), "sequence": record.get("sequence"),
                             "source": "verified_upgrade_rule", "reason": reason,
                             "original_action": record.get("action"),
                             "label_changed": record.get("action") != action}})
            counts["curated"] += 1
            if record.get("action") != action:
                counts["corrected"] += 1
    for name, rows in (("sft.jsonl", accepted), ("needs_review.jsonl", review)):
        (output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = dict(counts)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate verified upgrade corrections.")
    parser.add_argument("--input-dir", type=Path, default=Path("logs/strategic_trajectories"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/upgrade_corrections"))
    args = parser.parse_args()
    print(json.dumps(curate(args.input_dir, args.output_dir), ensure_ascii=False))
