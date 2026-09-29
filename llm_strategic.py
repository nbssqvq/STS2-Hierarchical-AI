from __future__ import annotations

import time
from typing import Any, Dict, Optional


class StrategicAgent:
    """Minimal deterministic fallback strategy that respects UI readiness."""

    def __init__(self):
        self._last_screen: Optional[str] = None
        self._last_action_name: Optional[str] = None
        self._selected_card_indices: Dict[str, list[int]] = {}
        self._screen_stuck_guard: Dict[str, Dict[str, Any]] = {}
        self._rest_choice_applied: Optional[tuple[int, int]] = None

    def _screen_name(self, state: Dict[str, Any]) -> str:
        if not isinstance(state, dict):
            return ""
        for key in ("screen", "state_type", "screen_type"):
            value = state.get(key)
            if value is not None:
                return str(value).strip().lower()
        return ""

    def _valid_option_indices(self, options: Any) -> list[int]:
        if not isinstance(options, list):
            return []
        valid: list[int] = []
        for idx, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            enabled = option.get("is_enabled")
            if enabled is None:
                enabled = option.get("enabled")
            if enabled is not None and enabled is False:
                continue
            if option.get("can_select") is False:
                continue
            if option.get("disabled") is True:
                continue
            valid.append(idx)
        if valid:
            return valid
        return list(range(len(options)))

    def _guard_stuck_option(self, screen: str, action_name: str, selected_index: int, options: Any = None) -> int:
        now = time.monotonic()
        previous = self._screen_stuck_guard.get(screen)
        valid_indices = self._valid_option_indices(options)

        if previous is not None and previous.get("action_name") == action_name and previous.get("index") == selected_index and now - float(previous.get("timestamp", now)) >= 10.0:
            if valid_indices:
                if selected_index in valid_indices:
                    current_position = valid_indices.index(selected_index)
                    next_position = (current_position + 1) % len(valid_indices)
                    fallback_index = valid_indices[next_position]
                else:
                    fallback_index = valid_indices[0]
                self._screen_stuck_guard[screen] = {"action_name": action_name, "index": fallback_index, "timestamp": now}
                return fallback_index

        self._screen_stuck_guard[screen] = {"action_name": action_name, "index": selected_index, "timestamp": now}
        return selected_index

    def _next_card_select_index(self, cards: list[Any], screen_name: str) -> int:
        selected = set(self._selected_card_indices.get(screen_name, []))
        for idx, _ in enumerate(cards):
            if idx not in selected:
                return idx
        if cards:
            return 0
        return 0

    def _record_selected_card_index(self, screen_name: str, index: int) -> None:
        selected = self._selected_card_indices.setdefault(screen_name, [])
        if index not in selected:
            selected.append(index)
        if len(selected) > 32:
            selected = selected[-32:]
        self._selected_card_indices[screen_name] = selected

    def _clear_card_selection_state(self, screen_name: str) -> None:
        self._selected_card_indices.pop(screen_name, None)

    def _card_selection_can_confirm(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        card_select_state = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
        if isinstance(card_select_state, dict):
            if card_select_state.get("can_confirm") is True:
                return True
            if card_select_state.get("preview_showing") is True:
                return True
        return bool(state.get("can_confirm"))

    def _card_selection_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        screen = self._screen_name(state)
        if screen == "card_reward":
            reward_state = state.get("card_reward", {}) if isinstance(state, dict) else {}
            cards = reward_state.get("cards", []) if isinstance(reward_state, dict) else []
            if isinstance(cards, list) and cards:
                self._last_screen = screen
                self._last_action_name = "select_card_reward"
                return {"action": "select_card_reward", "card_index": 0}
            if isinstance(reward_state, dict) and reward_state.get("can_skip") is True:
                self._last_screen = screen
                self._last_action_name = "skip_card_reward"
                return {"action": "skip_card_reward"}
            self._last_screen = screen
            self._last_action_name = "select_card_reward"
            return {"action": "select_card_reward", "card_index": 0}

        if isinstance(state, dict):
            card_select_state = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
            if not isinstance(card_select_state, dict):
                card_select_state = {}

            if self._card_selection_can_confirm(state):
                self._clear_card_selection_state(screen)
                self._last_screen = screen
                self._last_action_name = "confirm_selection"
                return {"action": "confirm_selection"}

            cards = card_select_state.get("cards", [])
            if isinstance(cards, list) and cards:
                selected = set(self._selected_card_indices.get(screen, []))
                selected_cards = card_select_state.get("selected_cards", [])
                if isinstance(selected_cards, list):
                    for entry in selected_cards:
                        if isinstance(entry, dict):
                            idx = entry.get("index", entry.get("card_index"))
                            if idx is not None:
                                try:
                                    selected.add(int(idx))
                                except (TypeError, ValueError):
                                    pass
                        elif isinstance(entry, int):
                            selected.add(entry)

                for idx, _ in enumerate(cards):
                    if idx not in selected:
                        self._record_selected_card_index(screen, idx)
                        self._last_screen = screen
                        self._last_action_name = "select_card"
                        return {"action": "select_card", "index": idx}

                chosen_index = self._next_card_select_index(cards, screen)
                self._record_selected_card_index(screen, chosen_index)
                self._last_screen = screen
                self._last_action_name = "select_card"
                return {"action": "select_card", "index": chosen_index}

        if state.get("can_proceed") is True:
            if screen in {"bundle_select"}:
                self._last_screen = screen
                self._last_action_name = "confirm_bundle_selection"
                return {"action": "confirm_bundle_selection"}
            self._last_screen = screen
            self._last_action_name = "proceed"
            return {"action": "proceed"}

        if screen in {"relic_select", "bundle_select"}:
            self._last_screen = screen
            self._last_action_name = "select_card"
            return {"action": "select_card", "index": 0}

        if screen in {"reward", "rewards"}:
            self._last_screen = screen
            self._last_action_name = "claim_reward"
            return {"action": "claim_reward", "index": 0}

        self._last_screen = screen
        self._last_action_name = "select_card"
        return {"action": "select_card", "index": 0}

    def _first_enabled_option_index(self, options: Any) -> Optional[int]:
        if not isinstance(options, list):
            return None
        for idx, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            enabled = option.get("is_enabled")
            if enabled is None:
                enabled = option.get("enabled")
            if enabled is True:
                return idx
        return None

    def _menu_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = state.get("menu_option") or state.get("option") or ""
        current = str(current).lower()
        if current in {"", "singleplayer", "main_menu"}:
            return {"action": "menu_select", "option": "singleplayer"}
        if current == "singleplayer":
            return {"action": "menu_select", "option": "standard"}
        if current == "standard":
            return {"action": "menu_select", "option": "IRONCLAD"}
        if current == "ironclad":
            return {"action": "menu_select", "option": "embark"}
        return {"action": "menu_select", "option": "singleplayer"}

    def _potion_inventory_full(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        if not isinstance(player, dict):
            return False
        max_slots = int(player.get("max_potion_slots", 0) or 0)
        if max_slots <= 0:
            return False
        potion_count = len(player.get("potions", []) or [])
        return potion_count >= max_slots

    def _is_potion_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        item_type = str(item.get("type", "") or item.get("kind", "") or item.get("category", "") or "").lower()
        item_name = str(item.get("name", "") or item.get("id", "") or "").lower()
        return "potion" in item_type or "potion" in item_name

    def _reward_choice_indices(self, state: Dict[str, Any]) -> list[int]:
        rewards_state = state.get("rewards", {}) if isinstance(state, dict) else {}
        items = rewards_state.get("items", []) if isinstance(rewards_state, dict) else []
        if not isinstance(items, list):
            return []

        if self._potion_inventory_full(state):
            return [idx for idx, item in enumerate(items) if not self._is_potion_item(item)]

        return list(range(len(items)))

    def _choose_route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        options = state.get("options", []) if isinstance(state, dict) else []
        if not isinstance(options, list) or not options:
            options = state.get("map", {}).get("nodes", []) if isinstance(state, dict) and isinstance(state.get("map"), dict) else []
        idx = self._first_enabled_option_index(options) if isinstance(options, list) else None
        if idx is None:
            idx = 0
        idx = self._guard_stuck_option(self._screen_name(state), "choose_map_node", idx, options)
        return {"action": "choose_map_node", "index": idx}

    def _shop_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("can_proceed") is True:
            return {"action": "proceed"}
        items = state.get("shop", {}).get("items", []) if isinstance(state.get("shop"), dict) else state.get("items", [])
        if isinstance(items, list):
            potions_full = self._potion_inventory_full(state)
            affordable = [
                item for item in items
                if isinstance(item, dict)
                and item.get("can_afford") is True
                and item.get("is_stocked") is not False
                and not (potions_full and self._is_potion_item(item))
            ]
            if affordable:
                return {"action": "shop_purchase", "index": int(affordable[0].get("index", 0))}
        return {"action": "proceed"}

    def _event_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        options = state.get("options", []) if isinstance(state, dict) else []
        idx = self._first_enabled_option_index(options)
        if idx is None:
            idx = 0
        idx = self._guard_stuck_option(self._screen_name(state), "choose_event_option", idx, options)
        return {"action": "choose_event_option", "index": idx}

    def _rest_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Choose one enabled rest option and leave only when the UI permits it."""
        screen = self._screen_name(state)
        nested = state.get("rest_site", {}) if isinstance(state, dict) and isinstance(state.get("rest_site"), dict) else {}
        options = nested.get("options", []) if isinstance(nested, dict) else state.get("options", [])
        can_proceed = nested.get("can_proceed", state.get("can_proceed", False)) if isinstance(nested, dict) else bool(state.get("can_proceed"))
        if can_proceed is True:
            self._last_screen = screen
            self._last_action_name = "proceed"
            return {"action": "proceed"}

        run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}
        visit_key = (int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0))
        if self._rest_choice_applied == visit_key:
            return {"action": "none"}

        index = self._first_enabled_option_index(options)
        if index is None and isinstance(options, list) and options:
            has_enabled_fields = any(
                isinstance(option, dict)
                and any(key in option for key in ("is_enabled", "enabled", "can_select", "disabled"))
                for option in options
            )
            if not has_enabled_fields:
                index = 0
        if index is None:
            return {"action": "none"}
        index = self._guard_stuck_option(screen, "choose_rest_option", index, options)
        self._last_screen = screen
        self._last_action_name = "choose_rest_option"
        return {"action": "choose_rest_option", "index": index}

    def note_action_result(self, state: Dict[str, Any], payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Remember successful rest choices only for the current act and floor."""
        if self._screen_name(state) not in {"rest", "rest_site", "campfire", "fire"}:
            return
        failed = response.get("status") == "error" or response.get("success") is False
        if failed or payload.get("action") != "choose_rest_option":
            return
        run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}
        self._rest_choice_applied = (int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0))

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return the deterministic fallback action for the current game screen."""
        screen = self._screen_name(state)

        if screen not in {"rest", "rest_site", "campfire", "fire"}:
            self._rest_choice_applied = None

        if screen in {"menu", "main_menu"}:
            return self._menu_decision(state)
        if screen in {"map", "map_screen"}:
            return self._choose_route(state)
        if screen in {"shop", "shop_screen"}:
            return self._shop_decision(state)
        if screen in {"event", "event_screen"}:
            return self._event_decision(state)
        if screen in {"rest", "rest_site", "campfire", "fire"}:
            return self._rest_decision(state)
        if screen in {"combat", "monster", "elite", "boss"}:
            raise RuntimeError("StrategicAgent.decide() cannot control combat reward profiles")

        if screen in {"reward", "rewards"}:
            rewards_state = state.get("rewards", {}) if isinstance(state, dict) else {}
            items = rewards_state.get("items", []) if isinstance(rewards_state, dict) else []
            if isinstance(items, list) and items:
                candidate_indexes = self._reward_choice_indices(state)
                if candidate_indexes:
                    idx = int(candidate_indexes[0])
                    idx = self._guard_stuck_option(screen, "claim_reward", idx, items)
                    self._last_screen = screen
                    self._last_action_name = "claim_reward"
                    return {"action": "claim_reward", "index": idx}
            if isinstance(rewards_state, dict) and rewards_state.get("can_proceed") is True:
                self._last_screen = screen
                self._last_action_name = "proceed"
                return {"action": "proceed"}
            self._last_screen = screen
            self._last_action_name = "proceed"
            return {"action": "proceed"}

        if screen in {"card_reward"}:
            reward_state = state.get("card_reward", {}) if isinstance(state, dict) else {}
            cards = reward_state.get("cards", []) if isinstance(reward_state, dict) else []
            if isinstance(cards, list) and cards:
                self._last_screen = screen
                self._last_action_name = "select_card_reward"
                return {"action": "select_card_reward", "card_index": 0}
            if isinstance(reward_state, dict) and reward_state.get("can_skip") is True:
                self._last_screen = screen
                self._last_action_name = "skip_card_reward"
                return {"action": "skip_card_reward"}
            self._last_screen = screen
            self._last_action_name = "select_card_reward"
            return {"action": "select_card_reward", "card_index": 0}

        if screen in {"card_select", "relic_select", "choose_a_card", "card_grid_selection"}:
            return self._card_selection_decision(state)

        if screen in {"bundle_select"}:
            bundle_state = state.get("bundle_select", {}) if isinstance(state, dict) else {}
            if isinstance(bundle_state, dict) and isinstance(bundle_state.get("bundles"), list) and bundle_state["bundles"]:
                if self._last_action_name == "select_bundle" and self._last_screen == screen:
                    self._last_screen = screen
                    self._last_action_name = "confirm_bundle_selection"
                    return {"action": "confirm_bundle_selection"}
                self._last_screen = screen
                self._last_action_name = "select_bundle"
                return {"action": "select_bundle", "index": 0}

        if screen in {"hand_select"}:
            hand_select = state.get("hand_select", {}) if isinstance(state.get("hand_select"), dict) else {}
            if bool(hand_select.get("can_confirm")) or bool(state.get("can_confirm")):
                self._last_screen = screen
                self._last_action_name = "combat_confirm_selection"
                return {"action": "combat_confirm_selection"}
            cards = hand_select.get("cards", state.get("cards", [])) if isinstance(state, dict) else []
            if isinstance(cards, list) and cards:
                self._last_screen = screen
                self._last_action_name = "combat_select_card"
                return {"action": "combat_select_card", "card_index": 0}
            self._last_screen = screen
            self._last_action_name = "proceed"
            return {"action": "proceed"}

        if screen in {"victory", "defeat", "game_over"}:
            return {"action": "menu_select", "option": "main_menu"}

        if state.get("can_proceed") is True:
            self._last_screen = screen
            self._last_action_name = "proceed"
            return {"action": "proceed"}

        self._last_screen = screen
        self._last_action_name = "proceed"
        return {"action": "proceed"}

FixedStrategicAgent = StrategicAgent


__all__ = ["StrategicAgent", "FixedStrategicAgent"]
