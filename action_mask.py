from typing import Dict

import numpy as np


def get_action_mask(state: Dict) -> np.ndarray:
    """Return a flattened mask compatible with MultiDiscrete([10, 5]).

    Environment callers refresh the HTTP state; dictionaries are explicit snapshots.
    Server can_play/unplayable_reason is authoritative, including dynamic card costs.
    Targets use the living-enemy order shared with PPOActionAdapter.
    """
    if hasattr(state, "get_state"):
        state = state.get_state()
    if not isinstance(state, dict):
        return np.zeros(15, dtype=np.int8)

    screen = str(state.get("state_type", "") or state.get("screen", "") or "").lower()
    if screen not in {"monster", "elite", "boss", "combat", "hand_select"}:
        return np.zeros(15, dtype=np.int8)

    card_mask = np.zeros(10, dtype=np.int8)
    target_mask = np.zeros(5, dtype=np.int8)

    player = state.get("player", {}) if isinstance(state, dict) else {}
    hand = player.get("hand", []) or []
    enemies = state.get("battle", {}).get("enemies", []) if isinstance(state, dict) else []

    alive_targets = [
        i for i, enemy in enumerate(enemies[:5])
        if isinstance(enemy, dict) and int(enemy.get("hp", 0) or 0) > 0
    ]
    if alive_targets:
        # PPOActionAdapter indexes the filtered living-enemy list.
        target_mask[:len(alive_targets)] = 1
    else:
        target_mask[0] = 1

    for card_idx, card in enumerate(hand[:10]):
        if not isinstance(card, dict):
            continue
        if not bool(card.get("can_play", False)) or card.get("unplayable_reason") not in (None, "", "None"):
            continue
        if card.get("target_type") in {"AnyEnemy", "AllEnemies", "RandomEnemy"} and not alive_targets:
            continue

        card_mask[card_idx] = 1

    return np.concatenate([card_mask, target_mask]).astype(np.int8)


__all__ = ["get_action_mask"]


if __name__ == "__main__":
    print(get_action_mask({
        "player": {"energy": 3, "hand": [{"cost": "1", "can_play": True, "target_type": "AnyEnemy"}]},
        "battle": {"enemies": [{"hp": 10}, {"hp": 8}]},
    }))

""" end of action_mask.py """



""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """ 


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """


""" end of action_mask.py """

