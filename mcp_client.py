import time
from typing import Any, Dict, Optional

import requests


class MCPClient:
    def __init__(self, base_url: str = "http://localhost:15526"):
        self.base_url = base_url.rstrip("/")
        self.state_url = f"{self.base_url}/api/v1/singleplayer"
        self.session = requests.Session()

    def get_state(self, max_wait: float = 20.0) -> Dict[str, Any]:
        """Read state with bounded retries for transient main-thread stalls."""
        deadline = time.monotonic() + max_wait
        last_error: Optional[requests.RequestException] = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if last_error is not None:
                    raise last_error
                raise TimeoutError(f"Timed out reading state from {self.state_url}")
            try:
                resp = self.session.get(self.state_url, timeout=min(5.0, remaining))
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected state format: {type(payload).__name__}")
                return payload
            except requests.RequestException as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.1, remaining))

    def post_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(self.state_url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response format: {type(data).__name__}")
        return data

    def wait_until_ready(self, max_wait: float = 15.0, poll_interval: float = 0.05) -> Dict[str, Any]:
        deadline = time.monotonic() + max_wait
        last_state: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            try:
                state = self.get_state(max_wait=min(5.0, max(deadline - time.monotonic(), 0.1)))
                last_state = state
                if state.get("actionable") is not False and state.get("state_type") not in {"unknown"}:
                    return state
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        if last_state is not None:
            return last_state
        raise TimeoutError(f"Timed out waiting for a ready state from {self.state_url}")

    def close(self):
        self.session.close()

    def menu_select(self, option: str) -> Dict[str, Any]:
        return self.post_action({"action": "menu_select", "option": option})

    def choose_map_node(self, index: int) -> Dict[str, Any]:
        return self.post_action({"action": "choose_map_node", "index": index})

    def choose_rest_option(self, index: int) -> Dict[str, Any]:
        return self.post_action({"action": "choose_rest_option", "index": index})

    def choose_event_option(self, index: int) -> Dict[str, Any]:
        return self.post_action({"action": "choose_event_option", "index": index})

    def shop_purchase(self, index: int) -> Dict[str, Any]:
        return self.post_action({"action": "shop_purchase", "index": index})

    def end_turn(self) -> Dict[str, Any]:
        return self.post_action({"action": "end_turn"})


def get_state(base_url: str = "http://localhost:15526") -> Dict[str, Any]:
    return MCPClient(base_url).get_state()


def post_action(payload: Dict[str, Any], base_url: str = "http://localhost:15526") -> Dict[str, Any]:
    return MCPClient(base_url).post_action(payload)
