from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from feature_extractor import FeatureExtractor
from action_mask import get_action_mask
from llm_strategic import StrategicAgent
from mcp_client import MCPClient
from ppo_adapter import PPOActionAdapter
from reward_function import RESERVED_REWARD_WEIGHTS, RewardCalculator, combat_reward_profile
from run_single_game import GameRunner
from strategic_trajectory import StrategicTrajectoryLogger


class STS2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url: str = "http://localhost:15526", strategic_agent: Optional[StrategicAgent] = None,
                 combat_stable_seconds: float = 0.0, trajectory_logger=None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.client = MCPClient(self.base_url)
        if combat_stable_seconds < 0.0:
            raise ValueError("combat_stable_seconds must be nonnegative")
        self.combat_stable_seconds = float(combat_stable_seconds)
        self.feature_extractor = FeatureExtractor()
        self.strategic_agent = strategic_agent or StrategicAgent()
        self.trajectory_logger = trajectory_logger
        if trajectory_logger is None and hasattr(self.strategic_agent, "decide_observation"):
            self.trajectory_logger = StrategicTrajectoryLogger()
        self.ppo_adapter = PPOActionAdapter(self)
        self.reward_calculator = RewardCalculator()
        self.reserved_reward_weights = dict(RESERVED_REWARD_WEIGHTS)
        self.action_space = spaces.MultiDiscrete([10, 5])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_extractor.TOTAL_DIM,),
            dtype=np.float32,
        )
        self.state: Dict[str, Any] = {}
        self.previous_state: Dict[str, Any] = {}
        self._reward_origin_state: Dict[str, Any] = {}
        self._last_action: Optional[Dict[str, Any]] = None
        self._last_action_screen: Optional[str] = None
        self._last_action_ts: float = 0.0
        self._screen_guard: Dict[str, Dict[str, Any]] = {}
        self._auto_truncated = False
        self._auto_truncation_reason = None
        self._recovering_run = False
        self._decision_snapshot = None
        self.snapshot_reuse_count = 0
        self.snapshot_expired_count = 0
        self._combat_log_state = None
        self._combat_reward_profile = None
        self._trajectory_episode_recorded = False


    def _auto_advance(self):
        """Consume all rule-controlled screens before returning an observation."""
        count = 0
        stalled_actions = 0
        deadline = time.monotonic() + 20.0
        attempts = {}
        last_signature = None
        stable_since = time.monotonic()
        last_log_key = None
        last_log_at = 0.0
        while not self.runner.is_combat(self.state) and not self.runner.is_terminal(self.state):
            if self._auto_truncated:
                return count
            screen = self.runner._screen_name(self.state)
            log_key = (screen, self.state.get('run', {}).get('floor', 0))
            if log_key != last_log_key or time.monotonic() - last_log_at >= 2.0:
                print(f"[AUTO] screen={screen}, floor={log_key[1]}", flush=True)
                last_log_key, last_log_at = log_key, time.monotonic()
            signature = json.dumps(self.state, sort_keys=True, default=str)
            if signature != last_signature:
                stable_since = time.monotonic()
                # Count stalled commands, not successful actions across several rooms.
                stalled_actions = 0
                deadline = stable_since + 20.0
                attempts.clear()
                last_signature = signature
            ready = screen not in {"rest", "rest_site", "campfire", "fire"} or time.monotonic() - stable_since >= 0.5
            response = None
            payload = self.runner.auto_advance(self.state) if ready else None
            if payload is not None:
                key = (signature, json.dumps(payload, sort_keys=True))
                sent, failed, sent_at = attempts.get(key, (0, True, 0.0))
                direct_choose = (payload.get("action") == "select_card"
                                 and self.state.get("card_select", {}).get("screen_type") == "choose")
                # Successful asynchronous commands wait for changed state. Failed
                # commands get one delayed retry against a freshly read state.
                retry_delay = 1.0 if direct_choose and not failed else 0.5
                if (failed or direct_choose) and sent < 2 and time.monotonic() - sent_at >= retry_delay:
                    response = self.client.post_action(payload)
                    failed = response.get("status") == "error" or response.get("success") is False
                    self.runner.note_action_result(self.state, payload, response)
                    # GET auto-opens a closed merchant inventory, so complete both
                    # phases before the next state read: close it, wait briefly for
                    # the proceed button, then send proceed once more.
                    shop_close_started = (
                        failed
                        and screen in {"shop", "shop_screen", "fake_merchant"}
                        and payload.get("action") == "proceed"
                        and response.get("error") == "No proceed button available or enabled"
                    )
                    if shop_close_started:
                        time.sleep(0.25)
                        response = self.client.post_action(payload)
                        failed = response.get("status") == "error" or response.get("success") is False
                        if not failed:
                            print("[AUTO_SHOP] inventory closed and shop exited", flush=True)
                    attempts[key] = (sent + 1 + int(shop_close_started), failed, time.monotonic())
                    if not failed and payload.get("action") == "select_card" and not direct_choose:
                        self.runner._selected_indices.add(payload["index"])
                    if failed:
                        print(f"[AUTO_RETRY] action={payload} response={response}", flush=True)
                        if payload.get("action") == "select_card":
                            self.runner._selected_indices.discard(payload.get("index"))
                        if payload.get("action") == "menu_select":
                            if payload.get("option") == "embark":
                                self.runner.embarking = False
                            elif payload.get("option") == "IRONCLAD":
                                self.runner.character_selected = False
                    count += 1
                    stalled_actions += 1
            # Give non-combat screens time to finish UI transitions and animations.
            time.sleep(min(0.2, 0.05 + (time.monotonic() - stable_since) * 0.05))
            self.state = self.client.wait_until_ready(max_wait=20.0)
            trace = self.runner.pending_strategic_trace
            if (trace is not None and payload is not None and response is not None
                    and trace.get("action") == payload):
                if self.trajectory_logger is not None:
                    self.trajectory_logger.record_decision(trace, response, self.state)
                self.runner.pending_strategic_trace = None
            self.max_floor_reached = max(self.max_floor_reached, int(self.state.get("run", {}).get("floor", 0) or 0))
            changed = json.dumps(self.state, sort_keys=True, default=str) != signature
            if not changed and not self.runner.is_combat(self.state) and not self.runner.is_terminal(self.state) and (stalled_actions > 20 or time.monotonic() > deadline):
                reason = "stalled_actions > 20" if stalled_actions > 20 else "no state progress for 20s"
                # The Mod cannot reset an active run. Do not emit a synthetic
                # episode boundary that DummyVecEnv immediately tries to reset.
                raise RuntimeError(f"[AUTO_STALLED] {reason}; total_actions={count}; "
                                   f"screen={screen}; return game to main menu before resuming; state={self.state}")
        return count

    def reset(self, *, seed=None, options=None):
        """Start a run and return its first actionable combat observation."""
        super().reset(seed=seed)
        self.runner = GameRunner(strategic_agent=self.strategic_agent,
                                 feature_extractor=self.feature_extractor,
                                 state_reader=self.client.get_state,
                                 trajectory_logger=self.trajectory_logger)
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.max_floor_reached = 0
        self._auto_truncated = False
        self._auto_truncation_reason = None
        self._ensure_main_menu()
        if self.trajectory_logger is not None:
            self.trajectory_logger.start_run()
        self._decision_snapshot = None
        # Recovery steps belong to the abandoned run, never the new episode.
        self.runner = GameRunner(strategic_agent=self.strategic_agent,
                                 feature_extractor=self.feature_extractor,
                                 state_reader=self.client.get_state,
                                 trajectory_logger=self.trajectory_logger)
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.max_floor_reached = 0
        self.previous_state = {}
        self._reward_origin_state = {}
        self._combat_log_state = None
        self._combat_reward_profile = None
        self._trajectory_episode_recorded = False
        self.state = self.client.wait_until_ready(max_wait=20.0)
        self._auto_advance()
        self._settle_policy_state()
        self._prepare_combat_reward()
        if self._auto_truncated:
            raise RuntimeError(f"Automatic advancement failed before first combat: {self._auto_truncation_reason}")
        if self.runner.is_terminal(self.state):
            raise RuntimeError(f"Run ended before its first combat: {self.state}")
        self.previous_state = self.state
        self._reward_origin_state = self.state
        return self.feature_extractor.extract(
            self.state, self._combat_reward_profile).astype(np.float32), {}

    def resume_current_run(self):
        """Attach to the currently open run without abandoning or resetting it."""
        self.runner = GameRunner(strategic_agent=self.strategic_agent,
                                 feature_extractor=self.feature_extractor,
                                 state_reader=self.client.get_state,
                                 trajectory_logger=self.trajectory_logger)
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.max_floor_reached = 0
        self.previous_state = {}
        self._reward_origin_state = {}
        self._combat_log_state = None
        self._combat_reward_profile = None
        self._trajectory_episode_recorded = False
        self._auto_truncated = False
        self._auto_truncation_reason = None
        self._decision_snapshot = None
        self.state = self.client.wait_until_ready(max_wait=20.0)
        if self.runner.is_terminal(self.state) or self.runner._screen_name(self.state) in {"menu", "main_menu"}:
            return self.reset()
        self.max_floor_reached = int(self.state.get("run", {}).get("floor", 0) or 0)
        self._auto_advance()
        self._settle_policy_state()
        self._prepare_combat_reward()
        self.previous_state = self.state
        self._reward_origin_state = self.state
        return self.feature_extractor.extract(
            self.state, self._combat_reward_profile).astype(np.float32), {"resumed": True}

    def _resolve_action_for_state(self, action):
        """Translate only actions sampled from a combat observation."""
        self.runner.observe_state(self.state)
        if not self.runner.is_combat(self.state):
            raise RuntimeError(f"PPO received non-combat observation: {self.state}")
        if not isinstance(action, dict) and not get_action_mask(self.state)[:10].any():
            return {"action": "end_turn"}
        return action if isinstance(action, dict) else self.ppo_adapter.translate(action)

    def _settle_policy_state(self):
        """Wait for a semantically actionable combat snapshot.

        Combat-start animations can keep changing draw-pile and hand fields for
        longer than 20 seconds on a busy client.  Requiring two completely equal
        JSON snapshots therefore treats normal animation progress as a stall.
        """
        deadline = time.monotonic() + 120.0
        while not self.runner.is_terminal(self.state):
            if self._auto_truncated:
                return
            if not self.runner.is_combat(self.state):
                self._auto_advance()
                continue

            battle = self.state.get("battle", {})
            player_phase = str(battle.get("player_phase", "")).lower()
            policy_ready = (player_phase not in {"start", "end", "ending"}
                            and battle.get("player_actions_disabled") is not True
                            and battle.get("hand_in_card_play") is not True)
            if policy_ready:
                if self.combat_stable_seconds > 0:
                    time.sleep(self.combat_stable_seconds)
                    self.state = self.client.get_state()
                    confirmed_battle = self.state.get("battle", {})
                    confirmed_phase = str(confirmed_battle.get("player_phase", "")).lower()
                    if (not self.runner.is_combat(self.state)
                            or confirmed_phase in {"start", "end", "ending"}
                            or confirmed_battle.get("player_actions_disabled") is True
                            or confirmed_battle.get("hand_in_card_play") is True):
                        continue
                self.runner.observe_state(self.state)
                return

            if time.monotonic() > deadline:
                raise RuntimeError(f"Combat observation did not stabilize: {self.state}")
            time.sleep(0.05)
            self.state = self.client.get_state()

    @staticmethod
    def _is_valid_action(payload, state):
        """Validate indices, server playability, and targets against the live state."""
        if not GameRunner.is_combat(state):
            return False
        if payload.get("action") != "play_card":
            return True
        index = payload.get("card_index")
        hand = state.get("player", {}).get("hand", [])
        if not isinstance(index, int) or not 0 <= index < len(hand):
            return False
        card = hand[index]
        if not card.get("can_play") or card.get("unplayable_reason") not in (None, "", "None"):
            return False
        living = [enemy.get("entity_id") for enemy in state.get("battle", {}).get("enemies", [])
                  if enemy.get("hp", 0) > 0]
        if card.get("target_type") == "AnyEnemy":
            return payload.get("target") in living
        if card.get("target_type") in {"AllEnemies", "RandomEnemy"}:
            return bool(living)
        return True

    @staticmethod
    def _recover_rejected_action(payload, state, error_message=""):
        """Rebuild one rejected combat action from a fresh server snapshot."""
        if not GameRunner.is_combat(state):
            return None
        if payload.get("action") != "play_card":
            return {"action": "end_turn"}
        index = payload.get("card_index")
        hand = state.get("player", {}).get("hand", [])
        if not isinstance(index, int) or not 0 <= index < len(hand):
            return {"action": "end_turn"}

        class StateView:
            pass

        view = StateView()
        view.state = state
        rebuilt = PPOActionAdapter(view).translate([index, 0])
        if ("requires a target" in str(error_message).lower()
                and rebuilt.get("action") == "play_card" and not rebuilt.get("target")):
            living = [enemy.get("entity_id") for enemy in state.get("battle", {}).get("enemies", [])
                      if isinstance(enemy, dict) and int(enemy.get("hp", 0) or 0) > 0
                      and enemy.get("entity_id")]
            if living:
                rebuilt["target"] = living[0]
        return rebuilt if STS2Env._is_valid_action(rebuilt, state) else {"action": "end_turn"}

    def step(self, action):
        """Record one combat transition and privately advance all intervening UI."""
        step_started = time.monotonic()
        reward_origin = self._reward_origin_state or self.state
        sampled_state = self.state
        snapshot = self._decision_snapshot
        self._decision_snapshot = None
        # A mask snapshot can become stale between prediction and POST even within
        # a few milliseconds. Always refresh at the execution boundary.
        if snapshot is not None:
            self.snapshot_expired_count += 1
        self.state = self.client.get_state()
        prev_state = self.state
        self._prepare_combat_reward()
        combat_profile = self._combat_reward_profile
        payload = {}
        response = {}
        if self.runner.is_combat(self.state):
            payload = self._resolve_action_for_state(action)
            index = payload.get("card_index")
            old_hand = sampled_state.get("player", {}).get("hand", [])
            new_hand = self.state.get("player", {}).get("hand", [])
            identity_changed = (payload.get("action") == "play_card"
                                and (not isinstance(index, int) or index < 0
                                     or index >= len(old_hand) or index >= len(new_hand)
                                     or old_hand[index].get("id") != new_hand[index].get("id")))
            if identity_changed or not self._is_valid_action(payload, self.state):
                print(f"[ACTION-INVALID] fallback=end_turn action={payload}", flush=True)
                payload = {"action": "end_turn"}
            response = self.client.post_action(payload)
        else:
            print(f"[STATE_DRIFT] sampled combat already left; screen={self.runner._screen_name(self.state)}", flush=True)
        if response.get("status") == "error" or response.get("success") is False:
            rejected_payload = payload
            fresh_state = self.client.get_state()
            retry_payload = self._recover_rejected_action(
                rejected_payload, fresh_state, response.get("error", ""),
            )
            if retry_payload is not None:
                self.state = fresh_state
                payload = retry_payload
                print(f"[ACTION_RECOVERED] rejected={rejected_payload} retry={retry_payload} "
                      f"error={response.get('error')}", flush=True)
                response = self.client.post_action(payload)
            if response.get("status") == "error" or response.get("success") is False:
                print(f"[ACTION_REJECTED] action={payload} response={response}", flush=True)
        self.state = self.client.wait_until_ready(max_wait=20.0)
        deadline = time.monotonic() + 20.0
        while (self.runner._screen_name(self.state) in self.runner.COMBAT_SCREENS
               and (not self.runner.is_combat(self.state)
                    or (payload.get("action") == "play_card"
                        and response.get("status") == "ok"
                        and self.state == prev_state)
                    or (payload.get("action") == "end_turn"
                        and response.get("status") == "ok"
                        and self.state.get("battle", {}).get("round") is not None
                        and self.state.get("battle", {}).get("round") == prev_state.get("battle", {}).get("round")))):
            if time.monotonic() > deadline:
                raise RuntimeError(f"Combat result did not settle: {self.state}")
            time.sleep(0.01)
            self.state = self.client.wait_until_ready(max_wait=20.0)
        combat_result = self.state
        combat_log_result = self._update_combat_log(prev_state, combat_result)
        self.previous_state = prev_state
        self._last_action = payload
        self.max_floor_reached = max(self.max_floor_reached, int(self.state.get("run", {}).get("floor", 0) or 0))
        self._auto_advance()
        self._settle_policy_state()
        if combat_log_result is None and self._combat_log_state is not None:
            combat_log_result = self._update_combat_log(prev_state, self.state)
        terminated = self.runner.is_terminal(self.state)
        # A terminal state may appear during automatic UI advancement or settling.
        # Evaluate the original formula once, using the actual terminal endpoint.
        reward_prev = reward_origin
        reward_curr = self.state
        combat_transition = None
        if combat_log_result is not None:
            combat_transition = {
                "type": combat_log_result["type"],
                "floor": combat_log_result["floor"],
                "profile": combat_profile,
                "victory": combat_log_result["victory"],
                "hp_lost": combat_log_result["hp_lost"],
                "potions_used": combat_log_result["potions_used"],
            }
        reward = self._compute_step_reward(reward_prev, reward_curr, combat_transition)
        reward_parts = self.reward_calculator.calculate_reward_breakdown(
            reward_prev, reward_curr, combat_transition)
        act_clears = self.reward_calculator.act_clear_count(reward_prev, reward_curr)
        if act_clears:
            print(f"[ACT_CLEAR] count={act_clears} bonus={act_clears * self.reward_calculator.act_clear_bonus:.2f} "
                  f"from={reward_prev.get('run')} to={reward_curr.get('run')}", flush=True)
        if combat_transition and any(reward_parts[key] for key in ("combat_win", "hp_loss", "potion_use")):
            print(f"[REWARD_BREAKDOWN] type={combat_transition['type']} floor={combat_transition['floor']} "
                  f"win={reward_parts['combat_win']:.2f} hp={reward_parts['hp_loss']:.2f} "
                  f"potion={reward_parts['potion_use']:.2f} act={reward_parts['act_clear']:.2f} "
                  f"terminal={reward_parts['victory'] + reward_parts['death']:.2f} "
                  f"total={reward_parts['total']:.2f}", flush=True)
        if combat_log_result is not None:
            self._combat_reward_profile = None
        self._prepare_combat_reward()
        if not terminated:
            self._reward_origin_state = self.state
        self.episode_steps += 1
        self.episode_reward += float(reward)
        info = {"max_floor": self.max_floor_reached, "auto_steps": self.runner.auto_step_count,
                "policy_screen": self.runner._screen_name(prev_state),
                "step_wall_seconds": time.monotonic() - step_started}
        if act_clears:
            info["act_clear_bonus"] = act_clears * self.reward_calculator.act_clear_bonus
        if combat_transition:
            info["reward_breakdown"] = reward_parts
        if combat_log_result is not None:
            info["combat_result"] = combat_log_result
        if self._auto_truncated:
            info["truncation_reason"] = self._auto_truncation_reason
        if terminated or self._auto_truncated:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_steps}
            if self.trajectory_logger is not None and not self._trajectory_episode_recorded:
                self.trajectory_logger.record_episode_end(
                    self.state, self.max_floor_reached, self.episode_steps, self.episode_reward)
                self._trajectory_episode_recorded = True
            if not self._recovering_run:
                print(f"[episode_debug] state={self.runner._screen_name(self.state)} floor={self.max_floor_reached} "
                      f"reward={self.episode_reward:.2f} combat_steps={self.episode_steps}", flush=True)
        observation = self.feature_extractor.extract(self.state, self._combat_reward_profile).astype(np.float32)
        return observation, float(reward), terminated, self._auto_truncated, info

    def _update_combat_log(self, combat_state, endpoint_state):
        """Create one result when a tracked combat ends."""
        combat_types = {"monster", "elite", "boss"}
        start_screen = self.runner._screen_name(combat_state)
        if self._combat_log_state is None and start_screen in combat_types:
            self._combat_log_state = {
                "type": start_screen,
                "start_hp": int(combat_state.get("player", {}).get("hp", 0) or 0),
                "start_potions": len(combat_state.get("player", {}).get("potions", []) or []),
                "floor": int(combat_state.get("run", {}).get("floor", 0) or 0),
                "turns": int(combat_state.get("battle", {}).get("round", 0) or 0),
            }
        if self._combat_log_state is None:
            return None
        self._combat_log_state["turns"] = max(
            self._combat_log_state["turns"],
            int(combat_state.get("battle", {}).get("round", 0) or 0),
            int(endpoint_state.get("battle", {}).get("round", 0) or 0),
        )
        endpoint_screen = self.runner._screen_name(endpoint_state)
        if endpoint_screen in combat_types or endpoint_screen in {
                "hand_select", "card_select", "card_grid_selection", "choose_a_card"}:
            return None
        current_hp = int(endpoint_state.get("player", {}).get(
            "hp", combat_state.get("player", {}).get("hp", 0)) or 0)
        result = {
            "type": self._combat_log_state["type"],
            "victory": endpoint_screen in {"rewards", "card_reward", "treasure", "map", "rest_site"},
            "hp_lost": max(0, self._combat_log_state["start_hp"] - current_hp),
            "potions_used": max(0, self._combat_log_state["start_potions"] - len(
                endpoint_state.get("player", {}).get("potions", []) or [])),
            "floor": self._combat_log_state["floor"],
            "turns": self._combat_log_state["turns"],
        }
        self._combat_log_state = None
        return result

    def _prepare_combat_reward(self):
        """Ask strategy once at battle entry and lock coefficients until it ends."""
        combat_type = GameRunner._screen_name(self.state)
        if combat_type not in {"monster", "elite", "boss"} or self._combat_reward_profile is not None:
            return
        suggested = None
        chooser = getattr(self.strategic_agent, "choose_combat_rewards", None)
        if callable(chooser) and not self._recovering_run:
            try:
                reward_context_state = dict(self.state)
                reward_context_state["strategic_context"] = self.runner.combat_reward_context(self.state)
                suggested = chooser(reward_context_state)
            except Exception as exc:
                if getattr(self.strategic_agent, "required", False):
                    raise
                print(f"[COMBAT_REWARD] chooser failed: {exc}; using defaults", flush=True)
        self._combat_reward_profile = combat_reward_profile(combat_type, suggested)
        if self._combat_log_state is None:
            self._combat_log_state = {
                "type": combat_type,
                "start_hp": int(self.state.get("player", {}).get("hp", 0) or 0),
                "start_potions": len(self.state.get("player", {}).get("potions", []) or []),
                "floor": int(self.state.get("run", {}).get("floor", 0) or 0),
                "turns": int(self.state.get("battle", {}).get("round", 0) or 0),
            }
        print(f"[COMBAT_REWARD] type={combat_type} floor={self.state.get('run', {}).get('floor', 0)} "
              f"coefficients={self._combat_reward_profile}", flush=True)

    def set_reward_weights(self, weights):
        """Reject the removed legacy switch instead of silently ignoring it."""
        raise RuntimeError(
            "set_reward_weights() is obsolete and does not configure rewards; "
            "use choose_combat_rewards() and the combat A/B/C/D profile"
        )

    def get_state(self):
        """Fetch the current game state for manual callers."""
        self.state = self.client.get_state()
        self._decision_snapshot = (self.state, time.monotonic())
        return self.state

    def _finish_active_run(self, state):
        """Drain the existing run through shared step/auto rules outside PPO."""
        self.state = state
        self._reward_origin_state = state
        self._recovering_run = True
        print(f"[RECOVER] finishing active run screen={self.runner._screen_name(state)}", flush=True)
        try:
            self._auto_advance()
            self._settle_policy_state()
            for index in range(5000):
                if self.runner.is_terminal(self.state):
                    print(f"[RECOVER] terminal={self.runner._screen_name(self.state)} turns={index}", flush=True)
                    return
                # End turns without defending so the old run reaches death quickly.
                # Calling the raw environment bypasses PPO/VecMonitor collection.
                _, _, terminated, truncated, _ = self.step({"action": "end_turn"})
                if truncated and not terminated:
                    raise RuntimeError("[RECOVER] active run truncated before terminal")
            if not self.runner.is_terminal(self.state):
                raise RuntimeError("[RECOVER] active run did not finish within 5000 turns")
        finally:
            self._recovering_run = False

    def _ensure_main_menu(self):
        """Wait for a confirmed main menu before starting the shared runner."""
        state = self.client.get_state()
        if self.runner._screen_name(state) == "menu" and state.get("menu_screen") == "main":
            return
        if not self.runner.is_terminal(state) and self.runner._screen_name(state) not in {"menu", "main_menu", "character_select"}:
            self._finish_active_run(state)
        response = self.client.menu_select("main_menu")
        if response.get("status") == "error" or response.get("success") is False:
            raise RuntimeError(f"Cannot reset active run through Mod API; return to main menu before retrying. response={response}")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = self.client.get_state()
            if self.runner._screen_name(state) == "menu" and state.get("menu_screen") == "main":
                return
            time.sleep(0.2)
        raise RuntimeError(f"Main menu did not appear: {state}")

    def _compute_step_reward(self, prev: Dict[str, Any], curr: Dict[str, Any], combat=None) -> float:
        if not prev:
            return 0.0
        if prev == curr and not combat:
            return 0.0
        # Screen names can stay unchanged while a battle loses HP or uses a
        # potion; the separate combat delta preserves those changes.
        return self.reward_calculator.calculate_reward(prev, curr, combat)

    def render(self, mode: str = "human"):
        return None

    def close(self):
        self.client.close()
        if hasattr(self.strategic_agent, "close"):
            self.strategic_agent.close()


__all__ = ["STS2Env"]
