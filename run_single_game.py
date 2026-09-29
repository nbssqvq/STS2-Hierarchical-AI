from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, Optional

from action_mask import get_action_mask
from feature_extractor import FeatureExtractor
from ppo_adapter import PPOActionAdapter
from strategic_routing import (DecisionKind, StrategicActionValidator, StrategicPolicyGuard,
                               StrategicObservationBuilder, StrategicRouter)


class GameRunner:
    """Shared rule dispatcher; auto_advance never calls a policy."""

    COMBAT_SCREENS = {"combat", "monster", "elite", "boss"}
    TERMINAL_SCREENS = {"game_over", "victory", "defeat", "lose", "victory_screen", "defeat_screen"}

    def __init__(self, strategic_agent=None, feature_extractor=None,
                 state_reader=None, trajectory_logger=None):
        """Track shared game progression and per-run strategic state."""
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.strategic_agent = strategic_agent
        self.router = StrategicRouter()
        self.observation_builder = StrategicObservationBuilder()
        self.last_strategic_observation = None
        self.state_reader = state_reader
        self.trajectory_logger = trajectory_logger
        self.pending_strategic_trace = None
        self.character_selected = False
        self.embarking = False
        self._resume_attempted = False
        self._resume_failed = False
        self.auto_step_count = 0
        self.seen_screens = set()
        self._selection_key = None
        self._selected_indices = set()
        self._shop_visit = None
        self._route_plan = None

    def _planned_map_action(self, state):
        """Follow a saved act route while the next node is still safe and reachable."""
        plan = self._route_plan
        if plan is None:
            return None
        run = state.get("run", {})
        if run.get("act") != plan["act"]:
            self._route_plan = None
            return None
        map_state = state.get("map", {})
        current = map_state.get("current_position") or {}
        position = (current.get("col"), current.get("row"))
        if position != plan["origin"]:
            if not plan["rooms"] or position != tuple(plan["rooms"][0][:2]):
                self._route_plan = None
                return None
            plan["origin"] = position
            plan["rooms"].pop(0)
        if not plan["rooms"]:
            self._route_plan = None
            return None
        target = plan["rooms"][0]
        player = state.get("player", {})
        hp_ratio = (float(player.get("hp") or 0)
                    / max(float(player.get("max_hp") or 1), 1))
        options = map_state.get("next_options", [])
        matching = next(((index, option) for index, option in enumerate(options)
                         if (option.get("col"), option.get("row")) == tuple(target[:2])), None)
        if matching is None:
            print("[ROUTE_REPLAN] planned node is not travelable", flush=True)
            self._route_plan = None
            return None
        index, option = matching
        safer = any(str(candidate.get("type", "")).lower() not in {"elite", "boss"}
                    for candidate in options if isinstance(candidate, dict))
        danger = str(option.get("type", "")).lower() in {"elite", "boss"}
        if (hp_ratio < plan["hp_ratio"] - 0.20 or (hp_ratio < 0.40 and danger and safer)):
            print(f"[ROUTE_REPLAN] hp_ratio={hp_ratio:.2f} planned={target}", flush=True)
            self._route_plan = None
            return None
        action = {"action": "choose_map_node", "index": int(option.get("index", index))}
        print(f"[ROUTE_FOLLOW] act={plan['act']} target={target} action={action} "
              f"plan_reason={plan['reason']}", flush=True)
        return action

    def _save_map_route(self, state, observation, decision, action):
        if action.get("action") != "choose_map_node":
            return
        candidates = observation.get("route_candidates", [])
        chosen = next((candidate for candidate in candidates
                       if candidate["id"] == decision.get("route_id")
                       and candidate["first_action_index"] == action.get("index")), None)
        if chosen is None:
            chosen = next((candidate for candidate in candidates
                           if candidate["first_action_index"] == action.get("index")), None)
        if chosen is None:
            return
        current = state.get("map", {}).get("current_position") or {}
        player = state.get("player", {})
        self._route_plan = {
            "act": state.get("run", {}).get("act"),
            "origin": (current.get("col"), current.get("row")),
            "rooms": [list(room) for room in chosen["rooms"]],
            "hp_ratio": float(player.get("hp") or 0) / max(float(player.get("max_hp") or 1), 1),
            "reason": str(decision.get("reason") or "").strip()[:80],
        }
        print(f"[ROUTE_PLAN] act={self._route_plan['act']} route_id={chosen['id']} "
              f"rooms={self._route_plan['rooms']} "
              f"reason={self._route_plan['reason']}", flush=True)

    def combat_reward_context(self, state):
        """Expose the already planned post-combat route to reward selection."""
        plan = self._route_plan
        if not isinstance(plan, dict):
            return {"planned_route_after_combat": [], "next_rest_steps": None}
        rooms = plan.get("rooms", [])
        # The first route node is the encounter currently being entered; later
        # nodes are the route decisions relevant to the upcoming battle.
        future_rooms = rooms[1:] if rooms else []
        route = [
            {"steps_ahead": index + 1, "type": str(room[2])}
            for index, room in enumerate(future_rooms)
            if isinstance(room, (list, tuple)) and len(room) >= 3
        ]
        rest_steps = next((item["steps_ahead"] for item in route
                           if str(item["type"]).lower() in {
                               "restsite", "rest_site", "rest", "campfire", "fire"
                           }), None)
        return {"planned_route_after_combat": route, "next_rest_steps": rest_steps,
                "route_reason": str(plan.get("reason", ""))[:80]}

    def _shop_context(self, state, screen):
        """Retain successful purchases across a multi-screen shop visit."""
        floor = state.get("run", {}).get("floor")
        if screen in {"shop", "shop_screen", "fake_merchant"}:
            key = (state.get("run", {}).get("act"), floor)
            if self._shop_visit is None or self._shop_visit.get("key") != key:
                self._shop_visit = {
                    "key": key,
                    "initial_gold": state.get("player", {}).get("gold"),
                    "purchases": [],
                    "card_removal_requested": False,
                }
            return {
                "initial_gold": self._shop_visit["initial_gold"],
                "purchases": [dict(item) for item in self._shop_visit["purchases"]],
                "card_removal_requested": self._shop_visit["card_removal_requested"],
            }
        if (screen in self.COMBAT_SCREENS or screen in self.TERMINAL_SCREENS
                or screen in {"map", "map_screen", "menu", "main_menu"}):
            self._shop_visit = None
        return None

    def note_action_result(self, state, payload, response):
        """Notify the policy about action outcomes and record successful shop actions."""
        screen = self._screen_name(state)
        failed = response.get("status") == "error" or response.get("success") is False
        if screen in {"menu", "main_menu"} and payload.get("action") == "menu_select":
            option = str(payload.get("option", "")).lower()
            if option == "continue":
                self._resume_failed = failed
                if not failed:
                    self._resume_attempted = True
            elif option == "abandon_run" and not failed:
                self._resume_attempted = False
                self._resume_failed = False
        if self.strategic_agent is not None and hasattr(self.strategic_agent, "note_action_result"):
            self.strategic_agent.note_action_result(state, payload, response)
        if failed or screen not in {"shop", "shop_screen", "fake_merchant"}:
            return
        if payload.get("action") != "shop_purchase":
            return
        self._shop_context(state, screen)
        shop = state.get("shop", {}) if isinstance(state.get("shop"), dict) else {}
        index = payload.get("index")
        item = next((item for position, item in enumerate(shop.get("items", []))
                     if isinstance(item, dict) and int(item.get("index", position)) == index), {})
        purchase = {key: item.get(key) for key in (
            "index", "category", "price", "card_name", "relic_name", "potion_name",
        ) if item.get(key) is not None}
        purchase.setdefault("index", index)
        if purchase not in self._shop_visit["purchases"]:
            self._shop_visit["purchases"].append(purchase)
        category = str(item.get("category") or item.get("type") or "").lower()
        if category == "card_removal":
            self._shop_visit["card_removal_requested"] = True

    @staticmethod
    def _screen_name(state):
        """Normalize the actual API state_type, with legacy aliases."""
        return str(state.get("state_type") or state.get("screen") or state.get("screen_type") or "").lower()

    @classmethod
    def is_terminal(cls, state):
        """Identify terminal screens without dismissing them."""
        return cls._screen_name(state) in cls.TERMINAL_SCREENS

    @classmethod
    def is_combat(cls, state):
        """Only expose actionable live combat to PPO."""
        message = str(state.get("message", "")).lower()
        return (cls._screen_name(state) in cls.COMBAT_SCREENS
                and state.get("actionable") is not False
                and state.get("battle", {}).get("turn", "player") == "player"
                and state.get("battle", {}).get("is_play_phase") is not False
                and "waiting for rewards" not in message
                and "combat ended" not in message)

    def observe_state(self, state):
        """Update stateful routing context whenever the controller sees a state."""
        return self.router.classify(state)

    @staticmethod
    def _first_option(options):
        """Choose the first enabled option using its API index."""
        for index, option in enumerate(options or []):
            if not isinstance(option, dict):
                return index
            if (option.get("enabled") is False or option.get("is_enabled") is False
                    or option.get("can_select") is False or option.get("disabled")
                    or option.get("is_locked")):
                continue
            return int(option.get("index", index))
        return None

    @staticmethod
    def _potion_inventory_full(state):
        """Return whether another shop potion cannot fit in the inventory."""
        player = state.get("player", {}) if isinstance(state, dict) else {}
        max_slots = int(player.get("max_potion_slots", 0) or 0)
        return max_slots > 0 and len(player.get("potions", []) or []) >= max_slots

    @staticmethod
    def _is_potion_item(item):
        """Identify a shop potion across current and legacy API fields."""
        return str(item.get("category") or item.get("type") or item.get("kind") or "").lower() == "potion"

    @staticmethod
    def _selection_required_count(selection):
        """Infer multi-select count when the Mod UI does not expose selected cards."""
        for key in ("required_count", "amount", "num_cards", "max_cards", "max_selection"):
            value = selection.get(key) if isinstance(selection, dict) else None
            if isinstance(value, int) and value > 0:
                return value
        prompt = str(selection.get("prompt", "") if isinstance(selection, dict) else "")
        match = re.search(r"(?:选择|select|choose)\s*(\d+)\s*(?:张|cards?)?", prompt, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    def _prepare_card_selection(self, state, screen):
        if screen not in {"card_select", "card_grid_selection", "choose_a_card"}:
            return None
        nested = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
        selection = state.get("card_select", nested)
        cards = selection.get("cards", state.get("cards", [])) if isinstance(selection, dict) else []
        key = (state.get("run", {}).get("floor"), selection.get("screen_type"),
               selection.get("prompt"), tuple(card.get("id") for card in cards if isinstance(card, dict)))
        if key != self._selection_key:
            self._selection_key = key
            self._selected_indices.clear()
        required = self._selection_required_count(selection)
        if required > 1 and len(self._selected_indices) >= required:
            return {"action": "confirm_selection"}
        return None

    def auto_advance(self, state):
        """Return one rule action, or None while waiting for a transition."""
        screen = self._screen_name(state)
        shop_context = self._shop_context(state, screen)
        decision_kind = self.router.classify(state)
        session = self.router.update_session(state, decision_kind)
        if screen not in {"card_select", "card_grid_selection", "choose_a_card"}:
            self._selection_key = None
            self._selected_indices.clear()
        selection_completion = self._prepare_card_selection(state, screen)
        if selection_completion is not None:
            return selection_completion
        self.seen_screens.add(screen)
        self.auto_step_count += 1
        if self.is_terminal(state) or self.is_combat(state):
            return None
        if state.get("actionable") is False or screen in {"unknown", ""}:
            return None
        if screen in self.COMBAT_SCREENS:
            return None  # The reward overlay appears asynchronously.
        if screen in {"map", "map_screen"} and decision_kind == DecisionKind.MECHANICAL:
            planned_map_action = self._planned_map_action(state)
            if planned_map_action is not None:
                return planned_map_action
        if decision_kind in {DecisionKind.STRATEGIC_DECISION, DecisionKind.COMBAT_INTERSTITIAL}:
            self.last_strategic_observation = self.observation_builder.build(state, self.router)
            if shop_context is not None:
                self.last_strategic_observation["shop_visit"] = shop_context
            if self.strategic_agent is not None:
                guarded_payload = StrategicPolicyGuard.action(state)
                if guarded_payload is not None:
                    print(f"[STRATEGY_GUARD] screen={screen} action={guarded_payload}", flush=True)
                    if StrategicActionValidator.validate(state, guarded_payload):
                        self.pending_strategic_trace = {
                            "screen": screen,
                            "floor": state.get("run", {}).get("floor"),
                            "observation": self.last_strategic_observation,
                            "action": dict(guarded_payload),
                            "source": "strategy_guard",
                            "proposed_action": None,
                            "reason": ("优先购买可支付的删牌" if screen in {"shop", "shop_screen", "fake_merchant"}
                                       else "当前生命不足，选择安全路线"),
                        }
                    return guarded_payload
                planned_map_action = (self._planned_map_action(state)
                                      if screen in {"map", "map_screen"} else None)
                if planned_map_action is not None:
                    return planned_map_action
                if session is not None and session.planned_actions:
                    decision = {"status": "act", "action": session.planned_actions.pop(0),
                                "reason": "continue_validated_plan"}
                elif (screen in {"map", "map_screen"}
                      and hasattr(self.strategic_agent, "plan_route")):
                    decision = self.strategic_agent.plan_route(self.last_strategic_observation, state)
                elif hasattr(self.strategic_agent, "decide_observation"):
                    decision = self.strategic_agent.decide_observation(self.last_strategic_observation, state)
                else:
                    decision = {"status": "act", "action": self.strategic_agent.decide(state),
                                "reason": "legacy_strategic_agent"}
                status = decision.get("status") if isinstance(decision, dict) else None
                decision_source = decision.get("source", "unknown") if isinstance(decision, dict) else "unknown"
                if status == "plan" and session is not None:
                    session.planned_actions = [item for item in decision.get("actions", []) if isinstance(item, dict)]
                    payload = session.planned_actions.pop(0) if session.planned_actions else None
                elif status == "finished":
                    nested_state = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
                    if nested_state.get("can_proceed") or state.get("can_proceed"):
                        payload = {"action": "proceed"}
                    else:
                        print(f"[LLM_INVALID] screen={screen} finished before proceed became available", flush=True)
                        payload = (self.strategic_agent.fallback_action(state)
                                   if hasattr(self.strategic_agent, "fallback_action") else None)
                        if payload is not None:
                            decision_source = "fallback"
                            print(f"[LLM_FALLBACK] screen={screen} action={payload}", flush=True)
                else:
                    payload = decision.get("action") if isinstance(decision, dict) else None
                payload = StrategicActionValidator.normalize(payload)
                if isinstance(payload, dict) and payload.get("action") == "none":
                    return None
                proposed_payload = dict(payload) if isinstance(payload, dict) else payload
                if (screen in {"card_select", "card_grid_selection", "choose_a_card"}
                        and isinstance(payload, dict) and payload.get("action") == "select_card"
                        and payload.get("index") in self._selected_indices):
                    screen_state = state.get(screen, {}) if isinstance(state.get(screen), dict) else {}
                    selection = state.get("card_select", screen_state)
                    cards = selection.get("cards", []) if isinstance(selection, dict) else []
                    replacement = next((int(card.get("index", position))
                                        for position, card in enumerate(cards)
                                        if isinstance(card, dict)
                                        and int(card.get("index", position)) not in self._selected_indices), None)
                    if replacement is not None:
                        print(f"[LLM_SELECTION_GUARD] repeated={payload.get('index')} replacement={replacement}",
                              flush=True)
                        payload = {"action": "select_card", "index": replacement}
                        decision_source = "selection_guard"
                if (payload is not None and getattr(self.strategic_agent, "requires_freshness_check", False)
                        and self.state_reader is not None):
                    current = self.state_reader()
                    if self.router.fingerprint(current) != self.router.fingerprint(state):
                        print("[LLM_STALE] state changed while the strategic decision was running", flush=True)
                        return None
                if payload is not None and not StrategicActionValidator.validate(state, payload):
                    print(f"[LLM_INVALID] screen={screen} action={payload}", flush=True)
                    if session is not None:
                        session.planned_actions.clear()
                        session.phase = "planning"
                    payload = None
                    if hasattr(self.strategic_agent, "fallback_action"):
                        fallback_payload = self.strategic_agent.fallback_action(state)
                        fallback_payload = StrategicActionValidator.normalize(fallback_payload)
                        if StrategicActionValidator.validate(state, fallback_payload):
                            payload = fallback_payload
                            decision_source = "fallback"
                            print(f"[LLM_FALLBACK] screen={screen} action={payload}", flush=True)
                if isinstance(payload, dict):
                    safer_event_action = StrategicPolicyGuard.safe_event_action(state, payload)
                    if (safer_event_action is not None
                            and StrategicActionValidator.validate(state, safer_event_action)):
                        print(f"[EVENT_HEALTH_GUARD] proposed={payload} "
                              f"chosen={safer_event_action}", flush=True)
                        payload = safer_event_action
                        decision_source = "event_health_guard"
                        if isinstance(decision, dict):
                            decision["reason"] = "支付生命后过于危险，改选无失血选项"
                if isinstance(payload, dict) and payload.get("action"):
                    if screen in {"map", "map_screen"} and payload.get("action") == "choose_map_node":
                        self._save_map_route(state, self.last_strategic_observation, decision, payload)
                    if session is not None:
                        session.phase = "finished" if payload.get("action") == "proceed" else "executing"
                        if session.phase == "finished":
                            session.owner = "mechanical"
                    self.pending_strategic_trace = {
                        "screen": screen,
                        "floor": state.get("run", {}).get("floor"),
                        "observation": self.last_strategic_observation,
                        "action": dict(payload),
                        "source": decision_source,
                        "proposed_action": proposed_payload,
                        "reason": str(decision.get("reason") or "").strip()[:80]
                        if isinstance(decision, dict) else "",
                    }
                    return payload
                # The strategic owner has not completed a valid decision. Do
                # not fall through to the legacy mechanical implementation,
                # especially when a proceed button happens to be enabled.
                return None
        if screen in {"menu", "main_menu", "character_select"}:
            menu = state.get("menu_screen", screen)
            if menu in {"main", "main_menu", "menu"}:
                self.character_selected = False
                self.embarking = False
                options = state.get("options", [])
                names = [item.get("name") if isinstance(item, dict) else item for item in options]
                if "continue" in names and not self._resume_failed and not self._resume_attempted:
                    option = "continue"
                    self._resume_attempted = True
                    print("[AUTO_MENU] resuming saved run with Continue", flush=True)
                elif "continue" in names and self._resume_attempted and not self._resume_failed:
                    # Continue was accepted; allow the loading transition to
                    # finish instead of issuing another menu command.
                    return None
                elif "abandon_run" in names:
                    option = "abandon_run"
                    print("[AUTO_MENU] abandoning saved run after Continue was unavailable or failed",
                          flush=True)
                else:
                    option = "singleplayer"
            elif menu == "popup":
                options = state.get("options", [])
                affirmative = {"yes", "confirm", "ok", "accept", "abandon", "confirm_abandon"}
                index = next((int(item.get("index", i)) for i, item in enumerate(options)
                              if isinstance(item, dict)
                              and str(item.get("name", "")).lower() in affirmative
                              and item.get("enabled", item.get("is_enabled", True)) is not False), None)
                if index is None:
                    index = next((i for i, item in enumerate(options)
                                  if isinstance(item, str) and item.lower() in affirmative), None)
                if index is None:
                    index = self._first_option(options)
                if index is None:
                    return None
                option = options[index].get("name") if isinstance(options[index], dict) else options[index]
                if str(option).lower() in affirmative:
                    print(f"[AUTO_MENU] confirming abandon dialog with {option}", flush=True)
            elif menu == "singleplayer":
                option = "standard"
            elif menu == "character_select":
                if self.embarking:
                    return None
                option = "embark" if self.character_selected else "IRONCLAD"
                self.embarking = self.character_selected
                self.character_selected = True
            else:
                raise RuntimeError(f"[AUTO] unhandled menu state={json.dumps(state)}")
            return {"action": "menu_select", "option": option}
        if screen == "hand_select":
            return PPOActionAdapter._hand_select_action(state)
        nested = state.get(screen, {})
        if not isinstance(nested, dict):
            nested = {}
        if screen in {"card_select", "card_grid_selection", "choose_a_card"}:
            selection = state.get("card_select", nested)
            # A choose screen commits one card directly; it is not a grid toggle.
            # Do not exhaust local selection memory while its animation is loading.
            if selection.get("screen_type") == "choose" or screen == "choose_a_card":
                index = self._first_option(selection.get("cards", state.get("cards", [])))
                return {"action": "select_card", "index": index} if index is not None else None
            if selection.get("can_confirm") or selection.get("preview_showing") or state.get("can_confirm"):
                return {"action": "confirm_selection"}
            cards = selection.get("cards", state.get("cards", []))
            selected = selection.get("selected_cards", [])
            selected_indices = {x.get("index", x.get("card_index")) if isinstance(x, dict) else x for x in selected}
            selected_indices.update(self._selected_indices)
            index = next((i for i, card in enumerate(cards)
                          if i not in selected_indices and not (isinstance(card, dict) and card.get("selected"))), None)
            if index is None:
                return None
            return {"action": "select_card", "index": index}
        if screen == "card_reward":
            if nested.get("cards"):
                return {"action": "select_card_reward", "card_index": 0}
            return {"action": "skip_card_reward"}
        if screen == "bundle_select":
            if nested.get("can_confirm") or nested.get("preview_showing") or nested.get("selected_bundles"):
                return {"action": "confirm_bundle_selection"}
            return {"action": "select_bundle", "index": 0}
        if screen in {"reward", "rewards"}:
            rewards = state.get("rewards", nested)
            player = state.get("player", {})
            full = len(player.get("potions", [])) >= int(player.get("max_potion_slots", 3))
            for i, item in enumerate(rewards.get("items", [])):
                if full and "potion" in str(item.get("type", item.get("kind", ""))).lower():
                    continue
                return {"action": "claim_reward", "index": i}
            return {"action": "proceed"}
        if screen in {"map", "map_screen"}:
            options = state.get("map", nested).get("next_options", [])
            index = self._first_option(options)
            return {"action": "choose_map_node", "index": index} if index is not None else None
        if screen in {"event", "event_screen", "dialogue"}:
            event = state.get("event", nested)
            if event.get("in_dialogue"):
                return {"action": "advance_dialogue"}
            options = event.get("options", state.get("options", []))
            if event.get("can_proceed") or state.get("can_proceed"):
                return {"action": "proceed"}
            if options:
                exits = [dict(item, index=item.get("index", i)) for i, item in enumerate(options)
                         if isinstance(item, dict) and item.get("is_proceed")]
                index = self._first_option(exits or options)
                return {"action": "choose_event_option", "index": index} if index is not None else None
            return None
        if screen in {"rest", "rest_site", "campfire", "fire"}:
            rest = state.get("rest_site", nested)
            if rest.get("can_proceed") or state.get("can_proceed"):
                return {"action": "proceed"}
            options = rest.get("options", state.get("options", []))
            if not options:
                return None
            index = self._first_option(options)
            return {"action": "choose_rest_option", "index": index} if index is not None else None
        if screen == "treasure":
            # The Mod opens the chest while building state; open_chest is unsupported.
            relics = nested.get("relics", [])
            if relics:
                return {"action": "claim_treasure_relic", "index": relics[0].get("index", 0)}
            return {"action": "proceed"} if nested.get("can_proceed") else None
        if screen in {"shop", "shop_screen", "fake_merchant"}:
            shop = state.get("shop", nested)
            if screen == "fake_merchant":
                shop = nested.get("shop", shop) if isinstance(nested, dict) else shop
            if not isinstance(shop, dict):
                return None
            items = shop.get("items", [])
            if not isinstance(items, list):
                return None
            potions_full = self._potion_inventory_full(state)
            affordable = [
                (int(item.get("index", position)), item)
                for position, item in enumerate(items)
                if isinstance(item, dict)
                and item.get("is_stocked") is not False
                and item.get("can_afford") is True
                and not (potions_full and self._is_potion_item(item))
            ]
            if affordable:
                index, item = min(affordable, key=lambda pair: pair[0])
                print(f"[AUTO_SHOP] buying index={index} category={item.get('category', '')} "
                      f"price={item.get('price', 0)}", flush=True)
                return {"action": "shop_purchase", "index": index}
            return {"action": "proceed"}
        if screen == "relic_select":
            return {"action": "select_relic", "index": 0}
        if screen == "crystal_sphere":
            if nested.get("can_proceed"):
                return {"action": "crystal_sphere_proceed"}
            if nested.get("tool") == "none":
                return {"action": "crystal_sphere_set_tool", "tool": "small"}
            cells = nested.get("clickable_cells", [])
            if cells:
                return {"action": "crystal_sphere_click_cell", "x": cells[0]["x"], "y": cells[0]["y"]}
            return None
        print(f"[AUTO] unhandled screen={screen} state={json.dumps(state)}", flush=True)
        raise RuntimeError(f"Unsupported non-combat screen: {screen}")

def heuristic_tactical_action(state):
    """Choose the first playable card for manual model-free runs."""
    for index, card in enumerate(state.get("player", {}).get("hand", [])[:10]):
        if card.get("can_play"):
            class StateView:
                """Expose the state to the existing action adapter."""
            view = StateView()
            view.state = state
            return PPOActionAdapter(view).translate([index, 0])
    return {"action": "end_turn"}


def run_single_game(model_path: Optional[str] = None, max_steps: int = 5000, verbose: bool = True,
                    resume_current: bool = False) -> Dict[str, Any]:
    """Evaluate via the exact same reset/step/auto_advance chain as training."""
    from sb3_contrib import MaskablePPO
    from sts2_env import STS2Env

    from ollama_strategic import create_strategic_agent
    from project_config import DEFAULT_PPO_MODEL

    model_path = str(model_path or DEFAULT_PPO_MODEL)
    model = MaskablePPO.load(model_path)
    env = STS2Env(strategic_agent=create_strategic_agent())
    if model_path:
        from mapping_checkpoint import load_mappings
        load_mappings(env.feature_extractor, model_path)
    try:
        obs, _ = env.resume_current_run() if resume_current else env.reset()
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            assert env.runner.is_combat(env.state), env.state
            if model is not None:
                action, _ = model.predict(obs, action_masks=get_action_mask(env), deterministic=False)
            else:
                action = heuristic_tactical_action(env.state)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if verbose:
                print(f"step={step} screen={env.runner._screen_name(env.state)} reward={reward:.2f}")
            if terminated or truncated:
                break
        else:
            raise RuntimeError(f"Evaluation did not finish within {max_steps} combat steps")
        return {"final_screen": env.runner._screen_name(env.state), "total_reward": total_reward,
                "steps": step, "max_floor_reached": env.max_floor_reached,
                "auto_steps": env.runner.auto_step_count, "final_state": env.state}
    finally:
        env.close()


def main():
    """Run one or more full evaluations."""
    parser = argparse.ArgumentParser()
    from project_config import DEFAULT_PPO_MODEL
    parser.add_argument("--model-path", default=str(DEFAULT_PPO_MODEL))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-failures", type=int, default=10,
                        help="For batch collection, recover and continue after this many failed attempts.")
    parser.add_argument("--resume-current", action="store_true")
    parser.add_argument("--require-ollama", action="store_true",
                        help="Exit before batch collection if the configured Ollama model is unavailable.")
    args = parser.parse_args()
    if args.require_ollama:
        from ollama_strategic import create_strategic_agent
        preflight_agent = create_strategic_agent()
        try:
            if not preflight_agent.model_available():
                raise RuntimeError(
                    f"Required Ollama model is unavailable: {preflight_agent.model}. "
                    "Start Ollama and verify the model before collecting trajectories."
                )
        finally:
            preflight_agent.close()
    results = []
    failures = 0
    attempts = 0
    while len(results) < args.episodes:
        attempts += 1
        try:
            results.append(run_single_game(
                args.model_path, args.max_steps,
                args.episodes == 1, args.resume_current and attempts == 1,
            ))
            if args.episodes > 1:
                print(f"[COLLECTION_PROGRESS] completed={len(results)}/{args.episodes} "
                      f"failures={failures}", flush=True)
        except Exception as exc:
            if args.episodes == 1:
                raise
            failures += 1
            print(f"[EPISODE_FAILED] attempt={attempts} failures={failures}/{args.max_failures} "
                  f"error={type(exc).__name__}: {exc}", flush=True)
            if failures >= args.max_failures:
                raise RuntimeError(
                    f"Batch collection stopped after {failures} failed attempts; "
                    f"completed={len(results)}/{args.episodes}"
                ) from exc
    print(json.dumps(results, ensure_ascii=True))


__all__ = ["GameRunner", "heuristic_tactical_action", "run_single_game"]

if __name__ == "__main__":
    main()
