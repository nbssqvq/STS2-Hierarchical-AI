from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Dict, Optional


class DecisionKind(str, Enum):
    """The owner of the next meaningful game decision."""

    TACTICAL_COMBAT = "tactical_combat"
    STRATEGIC_DECISION = "strategic_decision"
    COMBAT_INTERSTITIAL = "combat_interstitial"
    MECHANICAL = "mechanical"
    TERMINAL = "terminal"


@dataclass
class DecisionSession:
    """Track ownership until a screen's meaningful decisions are complete."""

    screen: str
    fingerprint: str
    owner: str
    context: DecisionKind
    phase: str = "planning"
    planned_actions: list[Dict[str, Any]] = field(default_factory=list)


class StrategicRouter:
    """Classify states while retaining whether a modal belongs to combat."""

    COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}
    COMBAT_MODAL_SCREENS = {"hand_select"}
    STRATEGIC_SCREENS = {
        "map", "map_screen", "shop", "shop_screen", "fake_merchant",
        "event", "event_screen", "rest", "rest_site", "campfire", "fire",
        "card_reward", "card_select", "card_grid_selection", "choose_a_card",
        "relic_select", "bundle_select",
    }
    TERMINAL_SCREENS = {
        "game_over", "victory", "defeat", "lose", "victory_screen", "defeat_screen",
    }

    def __init__(self) -> None:
        self.combat_session_active = False
        self.combat_origin_screen: Optional[str] = None
        self.session: Optional[DecisionSession] = None

    @staticmethod
    def _selectable(items: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        return [item for item in items if not isinstance(item, dict) or not (
            item.get("is_locked") is True or item.get("is_stocked") is False
            or item.get("can_afford") is False or item.get("can_select") is False
            or item.get("enabled") is False or item.get("is_enabled") is False
        )]

    @classmethod
    def _shop_selectable(cls, state: Dict[str, Any], items: Any) -> list[Any]:
        selectable = cls._selectable(items)
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        max_slots = int(player.get("max_potion_slots", 0) or 0)
        potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
        inventory_full = max_slots > 0 and len(potions) >= max_slots
        if not inventory_full:
            return selectable
        return [item for item in selectable if not (
            isinstance(item, dict)
            and str(item.get("category") or item.get("type") or "").lower() == "potion"
        )]

    def _has_meaningful_decision(self, state: Dict[str, Any], screen: str) -> bool:
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        if screen in {"map", "map_screen"}:
            container = state.get("map", nested)
            return bool(self._selectable(container.get("next_options", [])))
        if screen in {"shop", "shop_screen", "fake_merchant"}:
            container = state.get("shop", nested)
            return bool(self._shop_selectable(state, container.get("items", [])))
        if screen in {"event", "event_screen"}:
            return bool(self._selectable(nested.get("options", state.get("options", []))))
        if screen in {"rest", "rest_site", "campfire", "fire"}:
            container = state.get("rest_site", nested)
            return bool(self._selectable(container.get("options", state.get("options", []))))
        if screen == "card_reward":
            return bool(self._selectable(nested.get("cards", [])))
        if screen in {"card_select", "card_grid_selection", "choose_a_card"}:
            selection = state.get("card_select", nested)
            return not bool(selection.get("can_confirm") or selection.get("preview_showing")
                            or state.get("can_confirm")) and bool(self._selectable(selection.get("cards", [])))
        if screen == "relic_select":
            return not bool(nested.get("can_confirm")) and bool(
                self._selectable(nested.get("relics", nested.get("cards", []))))
        if screen == "bundle_select":
            return not bool(nested.get("can_confirm") or nested.get("selected_bundles")) and bool(
                self._selectable(nested.get("bundles", [])))
        return True

    def selectable_choice_count(self, state: Dict[str, Any], screen: Optional[str] = None) -> int:
        """Count actual alternatives; one alternative needs no language model."""
        screen = screen or self.screen_name(state)
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        if screen in {"map", "map_screen"}:
            container = state.get("map", nested)
            items = container.get("next_options", []) if isinstance(container, dict) else []
        elif screen in {"shop", "shop_screen", "fake_merchant"}:
            container = state.get("shop", nested)
            items = container.get("items", []) if isinstance(container, dict) else []
            selectable = self._shop_selectable(state, items)
            # Leaving without buying remains a meaningful alternative even when
            # exactly one item is affordable.
            return len(selectable) + (1 if selectable else 0)
        elif screen in {"event", "event_screen"}:
            items = nested.get("options", state.get("options", []))
        elif screen in {"rest", "rest_site", "campfire", "fire"}:
            container = state.get("rest_site", nested)
            items = container.get("options", state.get("options", [])) if isinstance(container, dict) else []
        elif screen == "card_reward":
            items = nested.get("cards", [])
            selectable = self._selectable(items)
            # Skipping remains a meaningful alternative to taking a lone card.
            return len(selectable) + (1 if selectable else 0)
        elif screen in {"card_select", "card_grid_selection", "choose_a_card"}:
            container = state.get("card_select", nested)
            items = container.get("cards", []) if isinstance(container, dict) else []
        elif screen == "relic_select":
            items = nested.get("relics", nested.get("cards", []))
        elif screen == "bundle_select":
            items = nested.get("bundles", [])
        elif screen == "hand_select":
            container = state.get("hand_select", nested)
            items = container.get("cards", state.get("cards", [])) if isinstance(container, dict) else []
        else:
            return 0
        return len(self._selectable(items))

    @staticmethod
    def screen_name(state: Dict[str, Any]) -> str:
        return str(state.get("state_type") or state.get("screen") or state.get("screen_type") or "").lower()

    def classify(self, state: Dict[str, Any]) -> DecisionKind:
        screen = self.screen_name(state)
        if screen in self.TERMINAL_SCREENS:
            self.combat_session_active = False
            self.combat_origin_screen = None
            return DecisionKind.TERMINAL
        if screen in self.COMBAT_SCREENS:
            self.combat_session_active = True
            self.combat_origin_screen = screen
            return DecisionKind.TACTICAL_COMBAT
        if screen in self.COMBAT_MODAL_SCREENS:
            # A battle payload is also sufficient when this is the first state
            # observed by a newly attached controller.
            in_combat = self.combat_session_active or isinstance(state.get("battle"), dict)
            hand_select = state.get("hand_select", {}) if isinstance(state.get("hand_select"), dict) else {}
            has_choice = not bool(hand_select.get("can_confirm") or state.get("can_confirm")) and bool(
                self._selectable(hand_select.get("cards", state.get("cards", []))))
            if not has_choice or self.selectable_choice_count(state, screen) <= 1:
                return DecisionKind.MECHANICAL
            return DecisionKind.COMBAT_INTERSTITIAL if in_combat else DecisionKind.STRATEGIC_DECISION
        if screen in self.STRATEGIC_SCREENS:
            # These states conclusively end the previous combat session. Card
            # selection spawned during combat uses hand_select instead.
            self.combat_session_active = False
            self.combat_origin_screen = None
            return (DecisionKind.STRATEGIC_DECISION
                    if (self._has_meaningful_decision(state, screen)
                        and self.selectable_choice_count(state, screen) > 1)
                    else DecisionKind.MECHANICAL)
        return DecisionKind.MECHANICAL

    @staticmethod
    def fingerprint(state: Dict[str, Any]) -> str:
        screen = StrategicRouter.screen_name(state)
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        candidates = (
            nested.get("items") or nested.get("cards") or nested.get("options")
            or nested.get("next_options")
            or state.get("options") or state.get("cards") or []
        )
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        data = {
            "screen": screen,
            "floor": state.get("run", {}).get("floor") if isinstance(state.get("run"), dict) else None,
            "prompt": nested.get("prompt"),
            "candidates": candidates,
            "hp": player.get("hp"),
            "gold": player.get("gold"),
        }
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def update_session(self, state: Dict[str, Any], kind: DecisionKind) -> Optional[DecisionSession]:
        if kind not in {DecisionKind.STRATEGIC_DECISION, DecisionKind.COMBAT_INTERSTITIAL}:
            self.session = None
            return None
        screen = self.screen_name(state)
        fingerprint = self.fingerprint(state)
        if (self.session is not None and self.session.screen == screen
                and self.session.phase == "executing" and self.session.planned_actions):
            # Expected state changes (gold, stock, selected cards) do not end a
            # multi-action transaction. Each remaining action is still
            # validated against the refreshed state before execution.
            self.session.fingerprint = fingerprint
        elif self.session is None or self.session.screen != screen or self.session.fingerprint != fingerprint:
            self.session = DecisionSession(
                screen=screen,
                fingerprint=fingerprint,
                owner="llm",
                context=kind,
            )
        return self.session


class StrategicObservationBuilder:
    """Create a compact, explicit input for the future Ollama agent."""

    @staticmethod
    def _cards(cards: Any) -> list[Dict[str, Any]]:
        if not isinstance(cards, list):
            return []
        result = []
        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            compact = {
                "index": card.get("index", index),
                "id": card.get("id"),
                "name": card.get("name"),
                "cost": card.get("cost"),
                "star_cost": card.get("star_cost"),
                "type": card.get("type"),
                "upgraded": card.get("is_upgraded"),
                "description": card.get("description"),
                "can_select": card.get("can_select", card.get("enabled")),
            }
            if "upgrade_preview" in card:
                compact["upgrade_preview"] = card["upgrade_preview"]
            result.append(compact)
        return result

    @staticmethod
    def _named(items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append({key: item.get(key) for key in (
                "id", "name", "description", "counter", "slot", "target_type",
            ) if item.get(key) is not None})
        return result

    @staticmethod
    def _options(items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        result = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                result.append({"index": position, "title": str(item)})
                continue
            result.append({key: item.get(key) for key in (
                "index", "id", "title", "name", "type", "description",
                "is_locked", "is_proceed", "enabled", "can_select",
            ) if item.get(key) is not None})
        return result

    @staticmethod
    def _items(items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        result = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            compact = {key: item.get(key) for key in (
                "index", "category", "price", "is_stocked", "can_afford", "on_sale",
                "card_id", "card_name", "card_type", "card_cost", "card_rarity",
                "card_description", "relic_id", "relic_name", "relic_description",
                "potion_id", "potion_name", "potion_description",
            ) if item.get(key) is not None}
            compact.setdefault("index", position)
            result.append(compact)
        return result

    @staticmethod
    def _map_state(value: Any) -> Dict[str, Any]:
        """Keep every reachable future route without unrelated map branches."""
        if not isinstance(value, dict):
            return {}

        def point(item: Any) -> Optional[list[Any]]:
            if not isinstance(item, dict):
                return None
            return [item.get("col"), item.get("row"), item.get("type")]

        graph = {}
        for item in value.get("nodes", []):
            if not isinstance(item, dict):
                continue
            children = [child for child in item.get("children", [])
                        if isinstance(child, list) and len(child) >= 2]
            graph[(item.get("col"), item.get("row"))] = [
                item.get("col"), item.get("row"), item.get("type"), children,
            ]

        current = value.get("current_position") or {}
        pending = [(current.get("col"), current.get("row"))] if isinstance(current, dict) else []
        reachable = set()
        while pending:
            position = pending.pop()
            if position in reachable:
                continue
            reachable.add(position)
            node = graph.get(position)
            if node is not None:
                pending.extend((child[0], child[1]) for child in node[3])
        nodes = [node for position, node in graph.items() if position in reachable]

        choices = []
        for position, item in enumerate(value.get("next_options", [])):
            if not isinstance(item, dict):
                continue
            leads_to = [candidate for candidate in (point(child) for child in item.get("leads_to", []))
                        if candidate is not None]
            choices.append([item.get("index", position), item.get("col"), item.get("row"),
                            item.get("type"), leads_to])

        bosses = []
        for boss in value.get("bosses", []):
            if isinstance(boss, dict):
                bosses.append([boss.get("col"), boss.get("row"), boss.get("id"), boss.get("name")])

        return {
            "format": {
                "nodes": "[col,row,type,children[[col,row],...]]",
                "choices": "[action_index,col,row,type,leads_to[[col,row,type],...]]",
                "points": "[col,row,type]",
                "bosses": "[col,row,id,name]",
            },
            "current": point(value.get("current_position")),
            "visited": [candidate for candidate in (point(item) for item in value.get("visited", []))
                        if candidate is not None],
            "choices": choices,
            "nodes": nodes,
            "bosses": bosses,
        }

    @staticmethod
    def _route_candidates(full_map: Dict[str, Any], hp_ratio: float, gold: int) -> list[Dict[str, Any]]:
        """Offer complete connected routes, with several risk profiles per choice."""
        graph = {(node[0], node[1]): node for node in full_map.get("nodes", [])}
        preferences = (
            {"RestSite": 5, "Shop": 2, "Unknown": 0 if hp_ratio < 0.6 else 1,
             "Monster": -2, "Elite": -9},
            {"RestSite": 4, "Shop": 4 if gold >= 150 else 1, "Unknown": 2,
             "Monster": -1, "Elite": -7 if hp_ratio < 0.7 else -3},
            {"RestSite": 3, "Shop": 2, "Unknown": 1, "Monster": -4, "Elite": -10},
        )
        routes = []
        seen = set()
        for choice in full_map.get("choices", []):
            if len(choice) < 4:
                continue
            start = (choice[1], choice[2])
            for weights in preferences:
                cache = {}

                def best(position):
                    if position in cache:
                        return cache[position]
                    node = graph.get(position)
                    if node is None:
                        return (0, [])
                    current = [node[0], node[1], node[2]]
                    tails = [best((child[0], child[1])) for child in node[3]
                             if child[1] > node[1]]
                    score, tail = max(tails, key=lambda result: result[0]) if tails else (0, [])
                    result = (weights.get(str(node[2]), 0) + score, [current] + tail)
                    cache[position] = result
                    return result

                _, path = best(start)
                signature = tuple((node[0], node[1]) for node in path)
                if not path or signature in seen:
                    continue
                seen.add(signature)
                routes.append({"id": len(routes), "first_action_index": choice[0], "rooms": path})
        def risky_elite(route: Dict[str, Any]) -> bool:
            # Conservative planning estimate, not a prediction of actual combat damage.
            projected_hp = hp_ratio
            for room in route["rooms"]:
                room_type = str(room[2]).lower()
                if room_type == "elite":
                    if projected_hp < 0.70:
                        return True
                    projected_hp -= 0.25
                elif room_type == "monster":
                    projected_hp -= 0.10
                elif room_type == "unknown":
                    projected_hp -= 0.03
                elif room_type in {"restsite", "rest_site", "rest", "campfire"}:
                    projected_hp = min(1.0, projected_hp + 0.25)
            return False

        if hp_ratio < 0.70:
            no_elite = [route for route in routes
                        if all(str(room[2]).lower() != "elite" for room in route["rooms"])]
            if no_elite:
                routes = no_elite
        else:
            safe_routes = [route for route in routes if not risky_elite(route)]
            if safe_routes:
                routes = safe_routes
        for index, route in enumerate(routes):
            route["id"] = index
        return routes

    def build(self, state: Dict[str, Any], router: StrategicRouter) -> Dict[str, Any]:
        screen = router.screen_name(state)
        kind = router.classify(state)
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        run = state.get("run", {}) if isinstance(state.get("run"), dict) else {}
        battle = state.get("battle", {}) if isinstance(state.get("battle"), dict) else {}
        selection = state.get("hand_select") if screen == "hand_select" else state.get("card_select")
        if not isinstance(selection, dict):
            selection = nested
        observation = {
            "decision_context": kind.value,
            "in_combat": kind == DecisionKind.COMBAT_INTERSTITIAL,
            "combat_type": router.combat_origin_screen,
            "screen": screen,
            "fingerprint": router.fingerprint(state),
            "allowed_actions": sorted(StrategicActionValidator.ACTIONS_BY_SCREEN.get(screen, set())),
            "run": {"act": run.get("act"), "floor": run.get("floor"), "ascension": run.get("ascension")},
            "player": {
                "character": player.get("character"), "hp": player.get("hp"),
                "max_hp": player.get("max_hp"), "gold": player.get("gold"),
                "energy": player.get("energy"), "max_energy": player.get("max_energy"),
                "relics": self._named(player.get("relics", [])),
                "potions": self._named(player.get("potions", [])),
                "deck": self._cards(player.get("deck", state.get("deck", []))),
                "hand": self._cards(player.get("hand", [])),
            },
            "prompt": selection.get("prompt", nested.get("prompt")),
            "screen_meta": {key: nested.get(key) for key in (
                "event_id", "event_name", "body", "prompt", "screen_type",
                "can_proceed", "can_skip", "can_confirm",
            ) if nested.get(key) is not None},
            "candidates": {
                "cards": self._cards(selection.get("cards", nested.get("cards", state.get("cards", [])))),
                "options": self._options(nested.get("options", state.get("options", []))),
                "items": self._items(nested.get("items", [])),
                "map_nodes": self._options(state.get("map", {}).get("next_options", []))
                if isinstance(state.get("map"), dict) else [],
            },
        }
        if (screen in {"card_select", "card_grid_selection"}
                and str(observation["screen_meta"].get("screen_type") or "").lower() == "upgrade"):
            from upgrade_catalog import enrich_upgrade_cards
            # The candidate list already includes the full current card text.
            # Keep the deck for synergy context without duplicating descriptions.
            observation["player"]["deck"] = [
                {key: card.get(key) for key in ("id", "name", "upgraded")}
                for card in observation["player"]["deck"]
            ]
            observation["candidates"]["cards"] = enrich_upgrade_cards(observation["candidates"]["cards"])
        if screen in {"map", "map_screen"}:
            observation["full_map"] = self._map_state(state.get("map"))
            # The map has no card-index action. A count preserves the complete
            # deck composition without repeating identical card descriptions.
            deck = {}
            for card in observation["player"]["deck"]:
                key = (card.get("id"), bool(card.get("upgraded")))
                if key not in deck:
                    deck[key] = {"id": key[0], "name": card.get("name"),
                                 "upgraded": key[1], "count": 0}
                deck[key]["count"] += 1
            observation["player"]["deck"] = list(deck.values())
            hp = player.get("hp") or 0
            max_hp = player.get("max_hp") or 1
            observation["route_candidates"] = self._route_candidates(
                observation["full_map"], hp / max_hp, int(player.get("gold") or 0),
            )
        if kind == DecisionKind.COMBAT_INTERSTITIAL:
            observation["combat_instruction"] = (
                "This choice was opened by the current combat and directly affects it. "
                "After the modal is completed, control returns to PPO."
            )
            observation["battle"] = {
                "round": battle.get("round"),
                "enemies": battle.get("enemies", []),
            }
        return observation


class StrategicActionValidator:
    """Reject actions that do not belong to the currently owned screen."""

    ACTIONS_BY_SCREEN = {
        "map": {"choose_map_node"}, "map_screen": {"choose_map_node"},
        "shop": {"shop_purchase", "proceed"}, "shop_screen": {"shop_purchase", "proceed"},
        "fake_merchant": {"shop_purchase", "proceed"},
        "event": {"choose_event_option", "advance_dialogue", "proceed"},
        "event_screen": {"choose_event_option", "advance_dialogue", "proceed"},
        "rest": {"choose_rest_option", "proceed"}, "rest_site": {"choose_rest_option", "proceed"},
        "campfire": {"choose_rest_option", "proceed"}, "fire": {"choose_rest_option", "proceed"},
        "card_reward": {"select_card_reward", "skip_card_reward"},
        "card_select": {"select_card", "confirm_selection"},
        "card_grid_selection": {"select_card", "confirm_selection"},
        "choose_a_card": {"select_card", "confirm_selection"},
        "relic_select": {"select_relic", "select_card", "confirm_selection"},
        "bundle_select": {"select_bundle", "confirm_bundle_selection"},
        "hand_select": {"combat_select_card", "combat_confirm_selection"},
    }

    @staticmethod
    def normalize(payload: Any) -> Any:
        """Convert the LLM's generic index into the Mod action contract."""
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        action = normalized.get("action")
        if action in {"select_card_reward", "combat_select_card"}:
            if "card_index" not in normalized and "index" in normalized:
                normalized["card_index"] = normalized.pop("index")
        elif action in {
            "choose_map_node", "shop_purchase", "choose_event_option", "choose_rest_option",
            "select_card", "select_relic", "select_bundle",
        }:
            if "index" not in normalized and "card_index" in normalized:
                normalized["index"] = normalized.pop("card_index")
        return normalized

    @classmethod
    def validate(cls, state: Dict[str, Any], payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        screen = StrategicRouter.screen_name(state)
        action = payload.get("action")
        if action not in cls.ACTIONS_BY_SCREEN.get(screen, set()):
            return False
        if action == "proceed" and screen in {"rest", "rest_site", "campfire", "fire"}:
            nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
            can_proceed = nested.get("can_proceed", state.get("can_proceed", False))
            if can_proceed is not True:
                return False
        index = payload.get("index", payload.get("card_index"))
        indexed_actions = {
            "choose_map_node", "shop_purchase", "choose_event_option", "choose_rest_option",
            "select_card_reward", "select_card", "select_relic", "select_bundle",
            "combat_select_card",
        }
        if action in indexed_actions and index is None:
            return False
        if index is not None and (not isinstance(index, int) or index < 0):
            return False
        if index is None:
            return True
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        if screen in {"map", "map_screen"}:
            container = state.get("map", nested)
            candidates = container.get("next_options", []) if isinstance(container, dict) else []
        elif screen in {"shop", "shop_screen", "fake_merchant"}:
            container = state.get("shop", nested)
            candidates = container.get("items", []) if isinstance(container, dict) else []
        elif screen == "hand_select":
            candidates = state.get("hand_select", nested).get("cards", [])
        elif screen == "card_reward":
            candidates = state.get("card_reward", nested).get("cards", [])
        elif screen in {"card_select", "card_grid_selection", "choose_a_card"}:
            candidates = state.get("card_select", nested).get("cards", [])
        else:
            candidates = nested.get("options", state.get("options", []))
        if isinstance(candidates, list) and candidates:
            allowed = {
                int(item.get("index", position)) if isinstance(item, dict) else position
                for position, item in enumerate(candidates)
            }
            if index not in allowed:
                return False
            selected = next((item for position, item in enumerate(candidates)
                             if (int(item.get("index", position)) if isinstance(item, dict) else position) == index), None)
            if isinstance(selected, dict):
                if selected.get("is_stocked") is False or selected.get("is_locked") is True:
                    return False
                if selected.get("can_afford") is False or selected.get("can_select") is False:
                    return False
                if selected.get("enabled") is False or selected.get("is_enabled") is False:
                    return False
                if screen in {"shop", "shop_screen", "fake_merchant"}:
                    player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
                    max_slots = int(player.get("max_potion_slots", 0) or 0)
                    potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
                    is_potion = str(selected.get("category") or selected.get("type") or "").lower() == "potion"
                    if is_potion and max_slots > 0 and len(potions) >= max_slots:
                        return False
            return True
        return True


class StrategicPolicyGuard:
    """Enforce deterministic hard rules before asking a language model."""

    SAFE_ROUTE_ORDER = {
        "rest": 0, "rest_site": 0, "restsite": 0, "campfire": 0, "fire": 0,
        "shop": 1, "event": 2, "unknown": 2, "monster": 3,
    }
    @staticmethod
    def _route_type(option: Dict[str, Any]) -> str:
        return str(option.get("type") or option.get("node_type") or option.get("room_type")
                   or option.get("name") or "").strip().lower()

    @classmethod
    def _shop_removal(cls, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Buy an available card removal before considering any other shop item."""
        shop = state.get("shop", {}) if isinstance(state.get("shop"), dict) else {}
        removals = [
            (position, item) for position, item in enumerate(shop.get("items", []))
            if isinstance(item, dict)
            and str(item.get("category") or item.get("type") or "").lower() == "card_removal"
            and item.get("is_stocked") is not False and item.get("can_afford") is True
        ]
        if not removals:
            return None
        position, removal = removals[0]
        return {"action": "shop_purchase", "index": int(removal.get("index", position))}

    @staticmethod
    def _event_hp_cost(option: Dict[str, Any]) -> int:
        description = str(option.get("description") or "")
        chinese = re.search(r"失去\s*(\d+)\s*点?生命(?!值上限)", description)
        damage = re.search(r"受到\s*(\d+)\s*点?伤害", description)
        english = re.search(r"lose\s*(\d+)\s*(?:hp|health)", description, re.IGNORECASE)
        match = chinese or damage or english
        return int(match.group(1)) if match else 0

    @staticmethod
    def _event_max_hp_growth(option: Dict[str, Any]) -> bool:
        text = " ".join(str(option.get(key) or "") for key in (
            "title", "name", "description", "effect",
        ))
        mentions_max_hp = re.search(
            r"最大生命|生命值上限|max(?:imum)?\s*(?:hp|health)|max_hp", text, re.IGNORECASE,
        )
        grants_max_hp = re.search(
            r"增加|提升|提高|获得|永久|每场战斗|gain|increase|raise|\+\s*\d+", text, re.IGNORECASE,
        )
        return bool(mentions_max_hp and grants_max_hp)

    @classmethod
    def safe_event_action(cls, state: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace a risky health trade only when a safe event option exists."""
        screen = StrategicRouter.screen_name(state)
        if screen not in {"event", "event_screen"} or payload.get("action") != "choose_event_option":
            return None
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        options = nested.get("options", state.get("options", []))
        if not isinstance(options, list):
            return None
        indexed = [(int(option.get("index", position)), option)
                   for position, option in enumerate(options) if isinstance(option, dict)]
        selected = next((option for index, option in indexed if index == payload.get("index")), None)
        if selected is None:
            return None
        cost = cls._event_hp_cost(selected)
        if cost <= 0:
            return None
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        hp = float(player.get("hp") or 0)
        max_hp = float(player.get("max_hp") or 0)
        if max_hp <= 0:
            return None
        gold_gain = "金币" in str(selected.get("description") or "")
        max_hp_growth = cls._event_max_hp_growth(selected)
        if max_hp_growth:
            # This is a soft stop for repeatable long-term trades: allow them
            # while healthy, then prefer immediate recovery/growth before HP
            # falls below 60% of max HP (or 30 HP, whichever is higher).
            stop_line = max(30.0, 0.60 * max_hp)
            unsafe = hp - cost <= 0 or hp / max_hp < 0.60 or hp - cost < stop_line
        else:
            unsafe = hp - cost <= 0 or (gold_gain and (hp - cost) / max_hp < 0.50)
        if not unsafe:
            return None
        safe = [(index, option) for index, option in indexed
                if cls._event_hp_cost(option) == 0 and option.get("is_locked") is not True
                and option.get("enabled") is not False and option.get("can_select") is not False]
        if not safe:
            return None
        if gold_gain:
            free_gold = [(index, option) for index, option in safe
                         if "金币" in str(option.get("description") or "")]
            if free_gold:
                safe = free_gold
            elif hp - cost > 0:
                # Avoid replacing a survivable gold trade with an unrelated
                # curse, card loss, or other unpriced event consequence.
                return None
        elif max_hp_growth:
            recovery_terms = ("恢复", "回复", "回血", "治愈", "heal", "restore", "recover")
            recovery = [(index, option) for index, option in safe
                        if any(term in " ".join(str(option.get(key) or "") for key in (
                            "title", "name", "description", "effect",
                        )).lower() for term in recovery_terms)]
            if recovery:
                safe = recovery
        return {"action": "choose_event_option", "index": safe[0][0]}

    @classmethod
    def action(cls, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        screen = StrategicRouter.screen_name(state)
        if screen in {"shop", "shop_screen", "fake_merchant"}:
            return cls._shop_removal(state)
        if screen not in {"map", "map_screen"}:
            return None
        player = state.get("player", {}) if isinstance(state.get("player"), dict) else {}
        hp = float(player.get("hp", 0) or 0)
        max_hp = float(player.get("max_hp", 0) or 0)
        if max_hp <= 0:
            return None
        hp_ratio = hp / max_hp
        container = state.get("map", {}) if isinstance(state.get("map"), dict) else {}
        options = StrategicRouter._selectable(container.get("next_options", []))
        typed = [(position, option, cls._route_type(option)) for position, option in enumerate(options)
                 if isinstance(option, dict)]
        if hp_ratio < 0.60:
            rest = [(position, option) for position, option, route_type in typed
                    if route_type in {"rest", "rest_site", "campfire", "fire", "restsite"}]
            if rest:
                position, option = rest[0]
                return {"action": "choose_map_node", "index": int(option.get("index", position))}
        if hp_ratio >= 0.70 or not any(route_type in {"elite", "boss"} for _, _, route_type in typed):
            return None
        safe = [(cls.SAFE_ROUTE_ORDER[route_type], position, option) for position, option, route_type in typed
                if route_type in cls.SAFE_ROUTE_ORDER]
        if not safe:
            return None
        _, position, option = min(safe, key=lambda entry: (entry[0], entry[1]))
        return {"action": "choose_map_node", "index": int(option.get("index", position))}


__all__ = ["DecisionKind", "DecisionSession", "StrategicActionValidator", "StrategicPolicyGuard",
           "StrategicObservationBuilder", "StrategicRouter"]
