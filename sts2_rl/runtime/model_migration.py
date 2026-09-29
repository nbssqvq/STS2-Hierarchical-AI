"""Migrate known observation layouts without changing learned outputs."""
import copy

import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import FlattenExtractor


def schema3_columns():
    """Map schema 2 into schema 3 with potion targets and card star costs."""
    columns = list(range(11))
    columns.extend(11 + slot * 2 for slot in range(5))
    for slot in range(10):
        old_start = 16 + slot * 8
        new_start = 21 + slot * 9
        columns.extend([new_start, new_start + 1])
        columns.extend(range(new_start + 3, new_start + 9))
    columns.extend(range(111, 212))
    if len(columns) != 197:
        raise AssertionError(f"Invalid schema-3 migration map: {len(columns)} columns")
    return columns


def migrate_schema3_model(model, target_observation_space):
    """Migrate schema 2 (197) to schema 3 (212), preserving outputs exactly."""
    old_dim = model.observation_space.shape[0]
    if old_dim == 212:
        return False
    if old_dim != 197 or target_observation_space.shape != (212,):
        raise ValueError(f"Unsupported schema-3 migration: {old_dim} -> {target_observation_space.shape}")
    if not isinstance(model.policy.features_extractor, FlattenExtractor):
        raise ValueError("Migration requires the existing flat MLP extractor")

    columns = schema3_columns()
    old_policy = model.policy
    weights = copy.deepcopy(old_policy.state_dict())
    optimizer = copy.deepcopy(old_policy.optimizer.state_dict())
    names = {id(parameter): name for name, parameter in old_policy.named_parameters()}
    parameter_names = [[names[id(parameter)] for parameter in group["params"]]
                       for group in old_policy.optimizer.param_groups]
    input_names = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}

    def expand(tensor):
        output = tensor.new_zeros((tensor.shape[0], 212))
        output[:, columns] = tensor
        return output

    for name in input_names:
        if name not in weights or weights[name].shape[1] != old_dim:
            raise ValueError(f"Unsupported input layer: {name}")
        weights[name] = expand(weights[name])
    for group, group_names in zip(optimizer["param_groups"], parameter_names):
        for parameter_id, name in zip(group["params"], group_names):
            if name in input_names:
                for key, value in optimizer["state"].get(parameter_id, {}).items():
                    if isinstance(value, torch.Tensor) and value.ndim == 2:
                        optimizer["state"][parameter_id][key] = expand(value)

    model.observation_space = target_observation_space
    model._setup_model()
    model.policy.load_state_dict(weights, strict=True)
    model.policy.optimizer.load_state_dict(optimizer)
    model._last_obs = None
    model._last_original_obs = None
    print("[MIGRATION] 197 -> 212; potion-target/star-cost inputs and optimizer moments zeroed", flush=True)
    return True


def migrate_combat_reward_features(model, target_observation_space):
    """Append four reward-profile inputs while preserving old model behavior."""
    old_dim = model.observation_space.shape[0]
    target_dim = target_observation_space.shape[0]
    if old_dim == target_dim == 216:
        return False
    if old_dim != 212 or target_dim != 216:
        raise ValueError(f"Unsupported reward-feature migration: {old_dim} -> {target_dim}")
    if not isinstance(model.policy.features_extractor, FlattenExtractor):
        raise ValueError("Migration requires the existing flat MLP extractor")

    old_policy = model.policy
    weights = copy.deepcopy(old_policy.state_dict())
    optimizer = copy.deepcopy(old_policy.optimizer.state_dict())
    names = {id(parameter): name for name, parameter in old_policy.named_parameters()}
    parameter_names = [[names[id(parameter)] for parameter in group["params"]]
                       for group in old_policy.optimizer.param_groups]
    input_names = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}

    def expand(tensor):
        output = tensor.new_zeros((tensor.shape[0], target_dim))
        output[:, :old_dim] = tensor
        return output

    for name in input_names:
        if name not in weights or weights[name].shape[1] != old_dim:
            raise ValueError(f"Unsupported input layer: {name}")
        weights[name] = expand(weights[name])
    for group, group_names in zip(optimizer["param_groups"], parameter_names):
        for parameter_id, name in zip(group["params"], group_names):
            if name in input_names:
                for key, value in optimizer["state"].get(parameter_id, {}).items():
                    if isinstance(value, torch.Tensor) and value.ndim == 2:
                        optimizer["state"][parameter_id][key] = expand(value)

    model.observation_space = target_observation_space
    model._setup_model()
    model.policy.load_state_dict(weights, strict=True)
    model.policy.optimizer.load_state_dict(optimizer)
    model._last_obs = None
    model._last_original_obs = None
    print("[MIGRATION] 212 -> 216; A/B/C/D inputs and optimizer moments zeroed", flush=True)
    return True


def load_for_training(path, env, hyperparameter_overrides=None):
    """Load with requested rollout settings, migrate, then attach the environment."""
    model = MaskablePPO.load(path, custom_objects=hyperparameter_overrides or {})
    from sts2_rl.runtime.feature_extractor import FeatureExtractor
    target_space = env.observation_space
    if model.observation_space.shape == (FeatureExtractor.TOTAL_DIM,):
        model.set_env(env)
        return model, False
    old_dim = model.observation_space.shape[0]
    migrated = False
    if old_dim == 197:
        intermediate = spaces.Box(-np.inf, np.inf, shape=(212,), dtype=np.float32)
        migrated = migrate_schema3_model(model, intermediate) or migrated
        old_dim = 212
    if old_dim == 212 and target_space.shape == (216,):
        migrated = migrate_combat_reward_features(model, target_space) or migrated
    elif old_dim != target_space.shape[0]:
        raise ValueError(f"Unsupported observation migration: {model.observation_space.shape} -> "
                         f"{target_space.shape}")
    model.set_env(env)
    return model, migrated
