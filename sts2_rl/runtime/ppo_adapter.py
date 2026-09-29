from typing import Dict, Sequence

import numpy as np


class PPOActionAdapter:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def _hand_select_action(state: Dict[str, object]) -> Dict[str, object]:
        hand_select = state.get("hand_select", {}) if isinstance(state, dict) else {}
        if not isinstance(hand_select, dict):
            hand_select = {}

        if bool(hand_select.get("can_confirm")) or bool(state.get("can_confirm")):
            return {"action": "combat_confirm_selection"}

        cards = hand_select.get("cards", [])
        if not isinstance(cards, list) or not cards:
            cards = state.get("cards", []) if isinstance(state, dict) else []
        if not isinstance(cards, list) or not cards:
            player = state.get("player", {}) if isinstance(state, dict) else {}
            cards = player.get("hand", []) or []
        if isinstance(cards, list) and cards:
            return {"action": "combat_select_card", "card_index": 0}
        return {"action": "end_turn"}

    def translate(self, action_indices: Sequence[int]) -> Dict[str, object]:
        state = getattr(self.env, "state", None)
        if state is None:
            state = self.env.get_state() if hasattr(self.env, "get_state") else {}

        screen = str(state.get("state_type", "") or state.get("screen", "") or "").lower()
        if screen == "hand_select":
            return self._hand_select_action(state)

        if isinstance(action_indices, np.ndarray):
            arr = action_indices.tolist()
        else:
            arr = list(action_indices)
        if len(arr) < 2:
            return {"action": "end_turn"}

        card_idx = int(arr[0])
        target_idx = int(arr[1])
        hand = state.get("player", {}).get("hand", []) if isinstance(state, dict) else []
        enemies = state.get("battle", {}).get("enemies", []) if isinstance(state, dict) else []
        if card_idx >= len(hand):
            return {"action": "end_turn"}

        card = hand[card_idx]
        if not isinstance(card, dict):
            return {"action": "end_turn"}

        payload: Dict[str, object] = {"action": "play_card", "card_index": card_idx}
        target_type = str(card.get("target_type", "Self") or "Self")
        target_key = target_type.lower().replace(" ", "")

        if target_key in {"self", "notarget", "none", ""}:
            return payload

        if target_key in {"anyenemy", "randomenemy"}:
            alive = [enemy for enemy in enemies if isinstance(enemy, dict) and int(enemy.get("hp", 0) or 0) > 0]
            if not alive:
                return {"action": "end_turn"}
            target = alive[min(target_idx, len(alive) - 1)]
            payload["target"] = target.get("entity_id") or target.get("name") or str(target.get("combat_id"))
            return payload

        if target_key in {"allenemies", "all_enemies"}:
            if any(isinstance(enemy, dict) and int(enemy.get("hp", 0) or 0) > 0 for enemy in enemies):
                payload["target"] = "all_enemies"
                return payload
            return {"action": "end_turn"}

        alive = [enemy for enemy in enemies if isinstance(enemy, dict) and int(enemy.get("hp", 0) or 0) > 0]
        if alive:
            payload["target"] = alive[min(target_idx, len(alive) - 1)].get("entity_id") or alive[0].get("name")
            return payload
        return {"action": "end_turn"}


__all__ = ["PPOActionAdapter"]
