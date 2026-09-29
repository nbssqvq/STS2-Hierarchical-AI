from typing import Dict

import numpy as np


class CardIdMap:
    def __init__(self):
        self.card_to_idx: Dict[str, int] = {}
        self.idx_to_card: Dict[int, str] = {}
        self.next_idx = 1

    def get_or_insert(self, card_id: str) -> int:
        if not card_id:
            return 0
        if card_id not in self.card_to_idx:
            self.card_to_idx[card_id] = self.next_idx
            self.idx_to_card[self.next_idx] = card_id
            self.next_idx += 1
        return self.card_to_idx[card_id]


class NameIdMap:
    EMPTY_ID = 0

    def __init__(self):
        self.name_to_idx: Dict[str, int] = {"": self.EMPTY_ID}
        self.idx_to_name: Dict[int, str] = {self.EMPTY_ID: ""}
        self.next_idx = 1

    def get_or_insert(self, name: str) -> int:
        key = str(name or "").strip()
        if not key:
            return self.EMPTY_ID
        if key not in self.name_to_idx:
            self.name_to_idx[key] = self.next_idx
            self.idx_to_name[self.next_idx] = key
            self.next_idx += 1
        return self.name_to_idx[key]

    def __len__(self) -> int:
        return self.next_idx


class FeatureExtractor:
    SCHEMA_VERSION = 4
    PLAYER_FEATURE_DIM = 10
    HAND_FEATURE_DIM = 9
    HAND_CAPACITY = 10
    ENEMY_FEATURE_DIM = 6
    MAX_ENEMIES = 5
    PLAYER_STATUS_DIM = 10
    PLAYER_NAME_DIM = 1
    MONSTER_NAME_SLOTS = MAX_ENEMIES
    MONSTER_NAME_DIM = 1
    POTION_SLOTS = 5
    POTION_FEATURE_DIM = 2
    GLOBAL_FEATURE_DIM = 6
    COMBAT_REWARD_FEATURE_DIM = 4
    PLAYER_STATUS_SLOTS = 10
    ENEMY_STATUS_SLOTS = 3
    STATUS_SLOT_DIM = 2
    DYNAMIC_STATUS_DIM = (PLAYER_STATUS_SLOTS + MAX_ENEMIES * ENEMY_STATUS_SLOTS) * STATUS_SLOT_DIM
    TOTAL_DIM = (
        PLAYER_FEATURE_DIM
        + HAND_CAPACITY * HAND_FEATURE_DIM
        + MAX_ENEMIES * ENEMY_FEATURE_DIM
        + PLAYER_STATUS_DIM
        + PLAYER_NAME_DIM
        + MONSTER_NAME_SLOTS * MONSTER_NAME_DIM
        + POTION_SLOTS * POTION_FEATURE_DIM
        + GLOBAL_FEATURE_DIM
        + DYNAMIC_STATUS_DIM
        + COMBAT_REWARD_FEATURE_DIM
    )

    ID_SCALE = 32.0
    GOLD_SCALE = 200.0

    @classmethod
    def _normalize_id(cls, value: int) -> float:
        """Map dynamic positive IDs to a stable bounded value; zero stays empty."""
        value = float(value or 0)
        return value / (value + cls.ID_SCALE) if value > 0 else 0.0

    @classmethod
    def _normalize_signed(cls, value: float, scale: float) -> float:
        """Bound signed numeric values while preserving their ordering and sign."""
        value = float(value or 0.0)
        return value / (scale + abs(value))

    # For the name dictionaries we follow the same dynamic-growth pattern as card IDs.
    # New roles / enemies / potions encountered during training are inserted lazily via get_or_insert().

    def __init__(self):
        self.card_map = CardIdMap()
        self.status_map = CardIdMap()
        self.player_name_map = NameIdMap()
        self.monster_name_map = NameIdMap()
        self.potion_name_map = NameIdMap()
        self.card_type_map = NameIdMap()
        self.card_attribute_map = NameIdMap()
        self.target_type_map = NameIdMap()
        self.intent_type_map = NameIdMap()

    def extract(self, state: Dict, combat_reward_profile: Dict = None) -> np.ndarray:
        vec = np.zeros(self.TOTAL_DIM, dtype=np.float32)
        idx = 0

        player = state.get("player", {}) if isinstance(state, dict) else {}
        hp = float(player.get("hp", 0) or 0)
        max_hp = float(player.get("max_hp", 1) or 1)
        energy = float(player.get("energy", 0) or 0)
        max_energy = float(player.get("max_energy", 3) or 3)
        block = float(player.get("block", 0) or 0)
        hand_size = len(player.get("hand", []) or [])
        draw_count = int(player.get("draw_pile_count", 0) or 0)
        discard_count = int(player.get("discard_pile_count", 0) or 0)
        exhaust_count = int(player.get("exhaust_pile_count", 0) or 0)
        potion_count = len(player.get("potions", []) or [])

        player_features = [
            hp / max_hp if max_hp > 0 else 0.0,
            energy / max_energy if max_energy > 0 else 0.0,
            block / max(max_hp, 1.0),
            hand_size / 10.0,
            draw_count / 30.0,
            discard_count / 30.0,
            exhaust_count / 20.0,
            potion_count / 5.0,
            max_hp / 200.0,
            max_energy / 10.0,
        ]
        vec[idx: idx + self.PLAYER_FEATURE_DIM] = np.asarray(player_features, dtype=np.float32)
        idx += self.PLAYER_FEATURE_DIM

        player_name = str(player.get("character") or player.get("name") or player.get("role") or "")
        player_name_idx = self.player_name_map.get_or_insert(player_name)
        vec[idx] = self._normalize_id(player_name_idx)
        idx += self.PLAYER_NAME_DIM

        potion_slots = np.zeros(self.POTION_SLOTS * self.POTION_FEATURE_DIM, dtype=np.float32)
        potions = player.get("potions", []) or []
        for i in range(self.POTION_SLOTS):
            potion = potions[i] if i < len(potions) else None
            if isinstance(potion, dict):
                potion_name = str(potion.get("name") or potion.get("id") or "")
                potion_name_idx = self.potion_name_map.get_or_insert(potion_name)
                potion_slots[i * self.POTION_FEATURE_DIM] = self._normalize_id(potion_name_idx)
                target_idx = self.target_type_map.get_or_insert(potion.get("target_type"))
                potion_slots[i * self.POTION_FEATURE_DIM + 1] = self._normalize_id(target_idx)
        potion_dim = self.POTION_SLOTS * self.POTION_FEATURE_DIM
        vec[idx: idx + potion_dim] = potion_slots
        idx += potion_dim

        hand = player.get("hand", []) or []
        for i in range(self.HAND_CAPACITY):
            card = hand[i] if i < len(hand) else None
            vec[idx: idx + self.HAND_FEATURE_DIM] = self._encode_card(card)
            idx += self.HAND_FEATURE_DIM

        battle = state.get("battle", {}) if isinstance(state, dict) else {}
        enemies = battle.get("enemies", []) or []
        for i in range(self.MAX_ENEMIES):
            enemy = enemies[i] if i < len(enemies) else None
            vec[idx: idx + self.ENEMY_FEATURE_DIM] = self._encode_enemy(enemy)
            idx += self.ENEMY_FEATURE_DIM

        monster_name_slots = np.zeros(self.MONSTER_NAME_SLOTS, dtype=np.float32)
        for i in range(self.MONSTER_NAME_SLOTS):
            enemy = enemies[i] if i < len(enemies) else None
            if isinstance(enemy, dict):
                enemy_name = str(enemy.get("name") or enemy.get("entity_id") or enemy.get("id") or "")
                enemy_name_idx = self.monster_name_map.get_or_insert(enemy_name)
                monster_name_slots[i] = self._normalize_id(enemy_name_idx)
        vec[idx: idx + self.MONSTER_NAME_SLOTS] = monster_name_slots
        idx += self.MONSTER_NAME_SLOTS

        status_map = {}
        for s in player.get("status", []) or []:
            if isinstance(s, dict):
                status_map[str(s.get("name", "")).strip()] = float(s.get("amount", 0) or 0)
        key_status = ["力量", "虚弱", "易伤", "人工制品", "敏捷", "金属化", "缓冲", "怒气", "鼓舞", "活力"]
        for status_name in key_status:
            vec[idx] = min(max(status_map.get(status_name, 0.0) / 10.0, 0.0), 1.0)
            idx += 1

        run_info = state.get("run", {}) if isinstance(state, dict) else {}
        act = int(run_info.get("act", 1) or 1)
        floor_num = int(run_info.get("floor", 0) or 0)
        ascension = int(run_info.get("ascension", 0) or 0)
        round_num = int(battle.get("round", 0) or 0)
        is_combat = 1.0 if str(state.get("state_type", "")).lower() in {"monster", "elite", "boss", "hand_select"} else 0.0
        gold = float(player.get("gold", 0) or 0)
        global_features = [
            act / 3.0,
            floor_num / 20.0,
            ascension / 20.0,
            round_num / 20.0,
            is_combat,
            self._normalize_signed(gold, self.GOLD_SCALE),
        ]
        vec[idx: idx + self.GLOBAL_FEATURE_DIM] = np.asarray(global_features, dtype=np.float32)
        idx += self.GLOBAL_FEATURE_DIM

        # Append to retain the positions of all pre-existing observation fields.
        player_status = self._encode_statuses(player.get("status"), self.PLAYER_STATUS_SLOTS)
        vec[idx:idx + player_status.size] = player_status
        idx += player_status.size
        for i in range(self.MAX_ENEMIES):
            enemy = enemies[i] if i < len(enemies) else None
            statuses = enemy.get("status") if isinstance(enemy, dict) else None
            encoded = self._encode_statuses(statuses, self.ENEMY_STATUS_SLOTS)
            vec[idx:idx + encoded.size] = encoded
            idx += encoded.size

        # Keep the previous observation layout intact and append the reward
        # context. Bounds match the validated A/B/C/D ranges used by rewards.
        profile = combat_reward_profile if isinstance(combat_reward_profile, dict) else {}
        vec[idx:idx + self.COMBAT_REWARD_FEATURE_DIM] = np.asarray([
            min(max(float(profile.get("A", 0.0) or 0.0) / 75.0, 0.0), 1.0),
            min(max(float(profile.get("B", 0.0) or 0.0) / 0.6, 0.0), 1.0),
            min(max(float(profile.get("C", 0.0) or 0.0) / 1.2, 0.0), 1.0),
            min(max(float(profile.get("D", 0.0) or 0.0) / 10.0, 0.0), 1.0),
        ], dtype=np.float32)
        idx += self.COMBAT_REWARD_FEATURE_DIM

        if idx != self.TOTAL_DIM:
            raise ValueError(f"Feature size mismatch: produced {idx}, expected {self.TOTAL_DIM}")
        return vec

    def _encode_statuses(self, statuses, capacity: int) -> np.ndarray:
        """Encode ordered status slots using persistent IDs and signed amounts.

        Zero IDs denote empty slots. Fixed transforms keep existing values stable
        as the dictionary grows; negative amounts remain distinguishable.
        """
        encoded = np.zeros((capacity, self.STATUS_SLOT_DIM), dtype=np.float32)
        slot = 0
        for status in statuses or []:
            if not isinstance(status, dict):
                continue
            key = str(status.get("id") or status.get("name") or "").strip()
            if not key:
                continue
            status_id = self.status_map.get_or_insert(key)
            # Register all observed statuses, even those beyond the slot limit.
            if slot >= capacity:
                continue
            amount = float(status.get("amount", 0) or 0)
            encoded[slot] = [self._normalize_id(status_id),
                             0.5 + 0.5 * amount / (10.0 + abs(amount))]
            slot += 1
        return encoded.ravel()

    def _encode_card(self, card: Dict) -> np.ndarray:
        """Encode ID, cost, star cost, type, playability, upgrade, two keywords, target."""
        if not isinstance(card, dict):
            return np.zeros(self.HAND_FEATURE_DIM, dtype=np.float32)
        cost = str(card.get("cost", "0") or "0").strip()
        cost_value = -1.0 if cost.upper() == "X" else float(cost)
        star_cost = str(card.get("star_cost", "0") or "0").strip()
        star_cost_value = -1.0 if star_cost.upper() == "X" else float(star_cost)
        keywords = []
        for item in card.get("keywords", []) or []:
            key = item.get("name", "") if isinstance(item, dict) else str(item)
            if key and key not in keywords:
                keywords.append(key)
        keyword_ids = [self.card_attribute_map.get_or_insert(k) for k in keywords]
        attributes = (keyword_ids + [0, 0])[:2]
        return np.asarray([
            self._normalize_id(self.card_map.get_or_insert(card.get("id") or card.get("name"))),
            self._normalize_signed(cost_value, 1.0),
            self._normalize_signed(star_cost_value, 1.0),
            self._normalize_id(self.card_type_map.get_or_insert(card.get("type") or card.get("card_type"))),
            float(bool(card.get("can_play", False))),
            float(bool(card.get("is_upgraded", card.get("upgraded", False)))),
            *[self._normalize_id(value) for value in attributes],
            self._normalize_id(self.target_type_map.get_or_insert(card.get("target_type"))),
        ], dtype=np.float32)

    def _encode_enemy(self, enemy: Dict) -> np.ndarray:
        """Encode HP/block ratios, first intent ID, and its damage."""
        if not isinstance(enemy, dict):
            return np.zeros(self.ENEMY_FEATURE_DIM, dtype=np.float32)
        max_hp = float(enemy.get("max_hp", 1) or 1)
        intents = enemy.get("intents", []) or []
        intent = intents[0] if intents and isinstance(intents[0], dict) else {}
        second_intent = intents[1] if len(intents) > 1 and isinstance(intents[1], dict) else {}
        return np.asarray([
            float(enemy.get("hp", 0) or 0) / max_hp,
            float(enemy.get("block", 0) or 0) / max_hp,
            self._normalize_id(self.intent_type_map.get_or_insert(intent.get("type") or intent.get("intent_type"))),
            self._normalize_signed(_parse_intent_damage(intent), 40.0),
            self._normalize_id(self.intent_type_map.get_or_insert(second_intent.get("type") or second_intent.get("intent_type"))),
            self._normalize_signed(_parse_intent_damage(second_intent), 40.0),
        ], dtype=np.float32)


def _parse_intent_damage(intent: Dict) -> float:
    if not isinstance(intent, dict):
        return 0.0
    if str(intent.get("type", "")).lower() == "attack":
        label = str(intent.get("label") or intent.get("value") or "0")
        cleaned = label.replace("×", "x").replace("X", "x")
        if "x" in cleaned.lower():
            left, right = cleaned.lower().split("x", 1)
            try:
                return float(left) * float(right)
            except ValueError:
                pass
        try:
            return float(cleaned)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


parse_intent_damage = _parse_intent_damage


__all__ = ["CardIdMap", "NameIdMap", "FeatureExtractor", "parse_intent_damage"]


if __name__ == "__main__":
    ex = FeatureExtractor()
    sample_state = {
        "state_type": "monster",
        "battle": {"round": 1, "enemies": [{"hp": 10, "max_hp": 20, "block": 0, "intents": [{"type": "Attack", "label": "3×2"}]}]},
        "player": {"hp": 20, "max_hp": 40, "energy": 3, "max_energy": 3, "block": 0, "gold": 30, "hand": [{"id": "STRIKE_IRONCLAD", "type": "Attack", "cost": "1", "target_type": "AnyEnemy", "can_play": True, "is_upgraded": False}], "status": [{"name": "力量", "amount": 2}], "potions": [{"id": "1"}]},
        "run": {"act": 1, "floor": 3, "ascension": 1},
    }
    print(ex.extract(sample_state).shape)
    print(ex.extract(sample_state)[:12])


""" end of feature_extractor.py """

