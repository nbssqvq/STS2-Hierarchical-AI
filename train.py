from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

_TENSORBOARD_AVAILABLE = importlib.util.find_spec("tensorboard") is not None
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from action_mask import get_action_mask
from ollama_strategic import create_strategic_agent
from sts2_env import STS2Env
from mapping_checkpoint import MappingCheckpointCallback, load_mappings, save_mappings
from model_migration import load_for_training
from combat_stats_callback import CombatStatsCallback


def _build_single_env(rank: int):
    env = STS2Env(strategic_agent=create_strategic_agent())
    env = ActionMasker(env, get_action_mask)
    return env


def _evaluate_model(model_path: str, episodes: int = 20, max_steps: int = 5000) -> tuple[float, float, float]:
    """Use the same single-game evaluator as the standalone evaluation command."""
    from evaluate_model import evaluate_model

    summary = evaluate_model(model_path, episodes=episodes, max_steps=max_steps)
    print(f"Evaluation: episodes={episodes}, avg_reward={summary['avg_reward']:.2f}, "
          f"avg_max_floor={summary['avg_max_floor']:.2f}, full_run_win_rate={summary['win_rate']:.2%}")
    return summary["avg_reward"], summary["avg_max_floor"], summary["win_rate"]


def make_env(rank: int):
    def _init():
        env = _build_single_env(rank)
        return env

    return _init


def train(total_timesteps: int = 20_000, n_envs: int = 1, model_path: str = "ppo_sts2_model",
          log_dir: str = "./logs", continue_from: str | None = None,
          save_freq: int = 10_000) -> MaskablePPO:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=str(Path(log_dir) / "episode"),
                         info_keywords=("max_floor", "auto_steps"))
    extractor = vec_env.venv.envs[0].unwrapped.feature_extractor
    if continue_from:
        load_mappings(extractor, continue_from)
    for wrapped_env in vec_env.venv.envs:
        wrapped_env.unwrapped.feature_extractor = extractor
    checkpoint_callback = MappingCheckpointCallback(
        extractor, save_freq=save_freq, save_path=str(Path(log_dir)), name_prefix="ppo_sts2"
    )
    training_callbacks = CallbackList([checkpoint_callback, CombatStatsCallback(log_interval=1000)])
    tb_log_dir = log_dir if _TENSORBOARD_AVAILABLE else None

    if continue_from:
        model, migrated = load_for_training(
            continue_from,
            vec_env,
            hyperparameter_overrides={
                "learning_rate": 0.0003,
                "n_steps": 1024,
                "batch_size": 128,
                "n_epochs": 8,
                "clip_range": 0.2,
            },
        )
        if migrated:
            migration_path = Path(log_dir) / "migrated_model"
            if migration_path.with_suffix(".zip").exists():
                raise FileExistsError(f"Use a new log directory: {migration_path}")
            model.save(str(migration_path))
            save_mappings(extractor, migration_path, model.num_timesteps)
        model.tensorboard_log = tb_log_dir
        model.verbose = 1
        formats = ["stdout", "csv"] + (["tensorboard"] if _TENSORBOARD_AVAILABLE else [])
        model.set_logger(configure(log_dir, formats))
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        model.set_env(vec_env)
        try:
            model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False,
                        progress_bar=False, callback=training_callbacks)
        except (Exception, KeyboardInterrupt):
            model.save(str(Path(log_dir) / "interrupted_model"))
            save_mappings(extractor, Path(log_dir) / "interrupted_model", model.num_timesteps)
            diagnostics = {
                "num_timesteps": model.num_timesteps,
                "metrics": {
                    key: float(value) for key, value in model.logger.name_to_value.items()
                    if key.startswith("train/")
                },
            }
            (Path(log_dir) / "interrupted_metrics.json").write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8"
            )
            raise
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            clip_range=0.2,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
            tensorboard_log=tb_log_dir,
        )
        try:
            model.learn(total_timesteps=total_timesteps, progress_bar=False,
                        callback=training_callbacks)
        except (Exception, KeyboardInterrupt):
            model.save(str(Path(log_dir) / "interrupted_model"))
            save_mappings(extractor, Path(log_dir) / "interrupted_model", model.num_timesteps)
            raise

    if tb_log_dir is None:
        print("TensorBoard logging disabled because tensorboard is not installed in the environment.")
    model.save(model_path)
    save_mappings(extractor, model_path, model.num_timesteps)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy for the STS2 RL environment.")
    parser.add_argument("--total-timesteps", type=int, default=20_000,
                        help="Total environment steps to train.")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel game environments.")
    parser.add_argument("--model-path", type=str, default="ppo_sts2_model",
                        help="Where to save the final model.")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="TensorBoard log directory.")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes after training.")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Optional checkpoint path to resume learning from a saved PPO model.")
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Autosave a PPO checkpoint every N training steps.")
    args = parser.parse_args()

    print(f"Starting PPO training for STS2: total_timesteps={args.total_timesteps}, "
          f"n_envs={args.n_envs}, model_path={args.model_path}, "
          f"continue_from={args.continue_from}, save_freq={args.save_freq}")
    model = train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        model_path=args.model_path,
        log_dir=args.log_dir,
        continue_from=args.continue_from,
        save_freq=args.save_freq,
    )
    print(f"Saved PPO model: {args.model_path}")
    print(f"TensorBoard logs: {args.log_dir}")
    _evaluate_model(args.model_path, episodes=args.eval_episodes)
    print(model)


if __name__ == "__main__":
    main()


__all__ = ["make_env", "train"]
