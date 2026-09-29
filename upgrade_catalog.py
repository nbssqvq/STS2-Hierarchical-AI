"""Verified card upgrade previews from the game's local wiki API.

The catalog is a fallback for a running Mod that predates upgrade_preview in
card_select. The live Mod preview always takes precedence when available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_CATALOG = Path(__file__).resolve().parent / "models" / "card_upgrade_catalog.json"
_cached_catalog: dict[str, dict[str, Any]] | None = None


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, dict[str, Any]]:
    global _cached_catalog
    if _cached_catalog is None:
        if not path.exists():
            _cached_catalog = {}
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            _cached_catalog = data.get("cards", {}) if isinstance(data, dict) else {}
    return _cached_catalog


def enrich_upgrade_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog = load_catalog()
    enriched = []
    for card in cards:
        item = dict(card)
        if "upgrade_preview" not in item and item.get("upgraded") is False:
            preview = catalog.get(str(item.get("id") or ""))
            if preview:
                item["upgrade_preview"] = preview
        enriched.append(item)
    return enriched


def _observed_upgrade_ids(input_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in input_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "strategic_decision":
                continue
            observation = row.get("observation", {})
            if observation.get("screen") not in {"card_select", "card_grid_selection"}:
                continue
            screen_type = str(observation.get("screen_meta", {}).get("screen_type") or "").lower()
            prompt = str(observation.get("prompt") or "").lower()
            if screen_type != "upgrade" and "升级" not in prompt and "upgrade" not in prompt:
                continue
            for card in observation.get("candidates", {}).get("cards", []):
                if isinstance(card, dict) and card.get("id"):
                    ids.add(str(card["id"]))
    return ids


def _fetch_upgrade(card_id: str, base_url: str) -> tuple[str, dict[str, Any] | None]:
    response = requests.get(f"{base_url.rstrip('/')}/api/v1/wiki",
                            params={"query": card_id, "type": "card", "limit": 1}, timeout=15)
    response.raise_for_status()
    matches = response.json().get("results", [])
    if not matches or matches[0].get("id") != card_id:
        return card_id, None
    upgraded = matches[0].get("upgraded")
    if not isinstance(upgraded, dict) or not upgraded.get("description"):
        return card_id, None
    return card_id, {key: upgraded.get(key) for key in ("cost", "star_cost", "description")}


def build_catalog(input_dir: Path, output: Path = DEFAULT_CATALOG,
                  base_url: str = "http://127.0.0.1:15526") -> dict[str, int]:
    cards: dict[str, dict[str, Any]] = {}
    if output.exists():
        cards = json.loads(output.read_text(encoding="utf-8")).get("cards", {})
    observed_ids = _observed_upgrade_ids(input_dir)
    ids = sorted(observed_ids - cards.keys())
    missing = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_upgrade, card_id, base_url): card_id for card_id in ids}
        for future in as_completed(futures):
            card_id = futures[future]
            try:
                _, preview = future.result()
            except (requests.RequestException, ValueError, KeyError):
                preview = None
            if preview is None:
                missing.append(card_id)
            else:
                cards[card_id] = preview
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": "local_game_wiki", "cards": dict(sorted(cards.items()))},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return {"observed_ids": len(observed_ids), "verified_previews": len(cards),
            "unresolved_ids": len(missing)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache verified card upgrade previews.")
    parser.add_argument("--input-dir", type=Path, default=Path("logs/strategic_trajectories"))
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    print(json.dumps(build_catalog(args.input_dir, args.output), ensure_ascii=False))
