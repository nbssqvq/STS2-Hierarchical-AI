from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any, Dict, Optional


class StrategicTrajectoryLogger:
    """Append-only strategic traces suitable for later SFT/preference building."""

    def __init__(self, root: str | Path = "logs/strategic_trajectories") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start_run()

    def start_run(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.path = self.root / f"{self.run_id}.jsonl"
        self.sequence = 0

    def _write(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["run_id"] = self.run_id
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def record_decision(self, trace: Dict[str, Any], response: Dict[str, Any],
                        after_state: Dict[str, Any]) -> None:
        self.sequence += 1
        failed = response.get("status") == "error" or response.get("success") is False
        source = str(trace.get("source", "unknown"))
        self._write({
            "record_type": "strategic_decision",
            "sequence": self.sequence,
            "screen": trace.get("screen"),
            "floor": trace.get("floor"),
            "observation": trace.get("observation"),
            "action": trace.get("action"),
            "proposed_action": trace.get("proposed_action"),
            "reason": trace.get("reason", ""),
            "source": source,
            # This only means the model produced an accepted action.  Strategic
            # quality is assigned later from rules and delayed outcomes.
            "sft_candidate": source in {"ollama", "shop_reserve_guard", "strategy_guard"} and not failed,
            "accepted": not failed,
            "response": response,
            "after": self._summary(after_state),
        })

    def record_episode_end(self, state: Dict[str, Any], max_floor: int,
                           combat_steps: int, reward: float) -> None:
        screen = str(state.get("state_type") or state.get("screen") or "").lower()
        self._write({
            "record_type": "episode_end",
            "final_screen": screen,
            "max_floor": max_floor,
            "combat_steps": combat_steps,
            "episode_reward": reward,
            "victory": screen in {"victory", "victory_screen"},
            "final": self._summary(state),
        })

    @staticmethod
    def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}
        return {
            "screen": str(state.get("state_type") or state.get("screen") or "").lower(),
            "act": run.get("act"),
            "floor": run.get("floor"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "gold": player.get("gold"),
            "deck_size": len(player.get("deck", [])) if isinstance(player.get("deck"), list) else None,
            "relic_count": len(player.get("relics", [])) if isinstance(player.get("relics"), list) else None,
            "potion_count": len(player.get("potions", [])) if isinstance(player.get("potions"), list) else None,
        }


__all__ = ["StrategicTrajectoryLogger"]
