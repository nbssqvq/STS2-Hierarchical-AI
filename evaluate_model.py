from __future__ import annotations

import argparse
from typing import Any, Dict, List

from run_single_game import run_single_game


def evaluate_model(model_path: str, episodes: int = 10, max_steps: int = 5000) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for _ in range(episodes):
        result = run_single_game(model_path=model_path, max_steps=max_steps, verbose=False)
        results.append(result)

    wins = sum(1 for item in results if str(item["final_screen"]).lower() in {"victory", "victory_screen"})
    avg_reward = sum(item["total_reward"] for item in results) / max(len(results), 1)
    avg_max_floor = sum(int(item.get("max_floor_reached", 0) or 0) for item in results) / max(len(results), 1)
    best_floor = max(int(item.get("max_floor_reached", 0) or 0) for item in results) if results else 0
    full_clear_count = sum(1 for item in results if str(item["final_screen"]).lower() in {"victory", "victory_screen"})
    summary = {
        "episodes": len(results),
        "wins": wins,
        "win_rate": wins / max(len(results), 1),
        "avg_reward": avg_reward,
        "avg_max_floor": avg_max_floor,
        "best_floor": best_floor,
        "full_clear_count": full_clear_count,
        "results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained STS2 PPO model across multiple runs.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained MaskablePPO model.")
    parser.add_argument("--episodes", type=int, default=10, help="How many full episodes to evaluate.")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps before aborting each run.")
    args = parser.parse_args()

    summary = evaluate_model(args.model_path, episodes=args.episodes, max_steps=args.max_steps)
    print(
        f"Evaluated {summary['episodes']} episodes: "
        f"wins={summary['wins']}, win_rate={summary['win_rate']:.2%}, "
        f"avg_reward={summary['avg_reward']:.2f}, avg_max_floor={summary['avg_max_floor']:.2f}, best_floor={summary['best_floor']}"
    )


if __name__ == "__main__":
    main()
