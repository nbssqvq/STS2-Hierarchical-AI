"""Reward terms for one PPO decision and its following automatic progression."""

import math
from types import MappingProxyType
from typing import Dict, Optional


COMBAT_BASE_REWARD = {"monster": 10.0, "elite": 25.0, "boss": 50.0}
COMBAT_POTION_COST = {"monster": 10.0, "elite": 2.0, "boss": 0.0}
COMBAT_A_BOUNDS = {"monster": (5.0, 20.0), "elite": (18.0, 40.0), "boss": (40.0, 75.0)}
COMBAT_C_BOUNDS = {"monster": (0.8, 1.2), "elite": (0.8, 1.2), "boss": (0.0, 0.1)}

# Reserved for future reward experiments. These terms are deliberately inactive
# in the current combat-profile formula; keeping them here documents their
# intended neutral defaults without exposing a mutable runtime switch.
RESERVED_REWARD_WEIGHTS = MappingProxyType({
    "hp_loss": 0.0,
    "max_hp_loss": 0.0,
    "gold_loss": 0.0,
    "card_loss": 0.0,
    "potion_use": 0.0,
})


def combat_reward_profile(combat_type: str, suggested: Optional[Dict] = None) -> Dict[str, float]:
    """Validate bounded LLM coefficients; D is fixed by encounter type for now."""
    if combat_type not in COMBAT_BASE_REWARD:
        raise ValueError(f"Unknown combat type: {combat_type}")
    defaults = {"A": COMBAT_BASE_REWARD[combat_type], "B": 0.5,
                "C": 0.0 if combat_type == "boss" else 1.0,
                "D": COMBAT_POTION_COST[combat_type]}
    bounds = {"A": COMBAT_A_BOUNDS[combat_type], "B": (0.4, 0.6),
              "C": COMBAT_C_BOUNDS[combat_type]}
    if isinstance(suggested, dict):
        for key, (low, high) in bounds.items():
            value = suggested.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                defaults[key] = max(low, min(high, float(value)))
    return defaults


class RewardCalculator:
    def __init__(self):
        """Initialize the fixed terminal and act-clear reward terms."""
        self.victory_bonus = 1500.0
        self.loss_penalty = 300.0
        self.act_clear_bonus = 60.0

    @staticmethod
    def act_clear_count(prev: Dict, curr: Dict) -> int:
        """Count act entries even when the Mod updates act and floor separately."""
        def effective_act(state: Dict) -> int:
            run = state.get("run", {}) or {}
            floor = int(run.get("floor", 0) or 0)
            reported = int(run.get("act", 1) or 1)
            floor_act = 1 + int(floor >= 18) + int(floor >= 34)
            return max(reported, floor_act)

        return max(0, effective_act(curr) - effective_act(prev))

    def calculate_reward(self, prev: Dict, curr: Dict,
                         combat: Optional[Dict] = None) -> float:
        """Calculate reward using only the active combat-profile path."""
        return self.calculate_reward_breakdown(prev, curr, combat)["total"]

    def calculate_reward_breakdown(self, prev: Dict, curr: Dict,
                                   combat: Optional[Dict] = None) -> Dict[str, float]:
        """Return each active reward term to make evaluation runs auditable."""
        if not isinstance(prev, dict) or not isinstance(curr, dict):
            return {"victory": 0.0, "death": 0.0, "act_clear": 0.0,
                    "combat_win": 0.0, "hp_loss": 0.0, "potion_use": 0.0, "total": 0.0}
        if prev == curr and not combat:
            return {"victory": 0.0, "death": 0.0, "act_clear": 0.0,
                    "combat_win": 0.0, "hp_loss": 0.0, "potion_use": 0.0, "total": 0.0}

        parts = {"victory": 0.0, "death": 0.0, "act_clear": 0.0,
                 "combat_win": 0.0, "hp_loss": 0.0, "potion_use": 0.0}
        prev_screen = str(prev.get("state_type") or prev.get("screen") or "").lower()
        curr_screen = str(curr.get("state_type") or curr.get("screen") or "").lower()
        if curr_screen in {"victory", "victory_screen"} and prev_screen not in {"victory", "victory_screen"}:
            parts["victory"] = self.victory_bonus
        if curr_screen in {"game_over", "defeat", "lose"} and prev_screen not in {"game_over", "defeat", "lose"}:
            parts["death"] = -self.loss_penalty

        parts["act_clear"] = self.act_clear_count(prev, curr) * self.act_clear_bonus

        if combat:
            profile = combat_reward_profile(combat["type"], combat.get("profile"))
            if combat.get("victory"):
                parts["combat_win"] = profile["A"] + profile["B"] * int(combat["floor"])
            parts["hp_loss"] = -profile["C"] * max(0.0, float(combat.get("hp_lost", 0)))
            parts["potion_use"] = -profile["D"] * max(0, int(combat.get("potions_used", 0)))
        parts["total"] = sum(parts.values())
        return parts

    def __call__(self, prev: Dict, curr: Dict,
                 combat: Optional[Dict] = None) -> float:
        """Delegate callable use to the active reward calculation."""
        return self.calculate_reward(prev, curr, combat)


__all__ = ["RESERVED_REWARD_WEIGHTS", "RewardCalculator", "combat_reward_profile"]
