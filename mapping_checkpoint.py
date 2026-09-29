"""Checkpoint-specific persistence for environment observation dictionaries."""
import json
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

MAPS = {
    "card_map": ("card_to_idx", "idx_to_card"),
    "status_map": ("card_to_idx", "idx_to_card"),
    "player_name_map": ("name_to_idx", "idx_to_name"),
    "monster_name_map": ("name_to_idx", "idx_to_name"),
    "potion_name_map": ("name_to_idx", "idx_to_name"),
    "card_type_map": ("name_to_idx", "idx_to_name"),
    "card_attribute_map": ("name_to_idx", "idx_to_name"),
    "target_type_map": ("name_to_idx", "idx_to_name"),
    "intent_type_map": ("name_to_idx", "idx_to_name"),
}


def mapping_path(model_path):
    """Return the sidecar path for a checkpoint, with or without .zip."""
    path = Path(model_path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    return Path(str(path) + ".mappings.json")


def save_mappings(extractor, model_path, num_timesteps):
    """Atomically save all dictionaries including their allocation counters."""
    data = {"version": 2, "feature_schema": extractor.SCHEMA_VERSION,
            "num_timesteps": int(num_timesteps), "maps": {}}
    for name, (forward, _) in MAPS.items():
        mapping = getattr(extractor, name)
        data["maps"][name] = {"entries": dict(getattr(mapping, forward)),
                              "next_idx": mapping.next_idx}
    path = mapping_path(model_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_mappings(extractor, model_path):
    """Restore a paired sidecar; explicitly report legacy checkpoints."""
    path = mapping_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint dictionaries: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    saved_schema = data.get("feature_schema")
    compatible_schema = saved_schema in {2, 3, extractor.SCHEMA_VERSION}
    if data.get("version") != 2 or not compatible_schema:
        raise ValueError(f"Old feature encoding is incompatible; start a new model: {path}")
    for name, (forward, reverse) in MAPS.items():
        saved = data["maps"][name]
        entries, next_idx = saved["entries"], saved["next_idx"]
        if (not isinstance(next_idx, int) or next_idx < 1
                or any(type(i) is not int or not 0 <= i < next_idx
                       or (i == 0 and key != "") for key, i in entries.items())
                or len(set(entries.values())) != len(entries)):
            raise ValueError(f"Invalid dictionary {name}: {path}")
        mapping = getattr(extractor, name)
        setattr(mapping, forward, dict(entries))
        setattr(mapping, reverse, {i: key for key, i in entries.items()})
        mapping.next_idx = next_idx
    print(f"[MAPPINGS] loaded {path}: " + str({name: len(getattr(getattr(extractor, name), fields[0])) for name, fields in MAPS.items()}), flush=True)
    return True


class MappingCheckpointCallback(CheckpointCallback):
    """Save the environment dictionaries alongside each PPO checkpoint."""

    def __init__(self, extractor, **kwargs):
        """Keep the shared environment extractor for serialization."""
        super().__init__(**kwargs)
        self.extractor = extractor

    def _on_step(self):
        """Pair the standard checkpoint with its current dictionaries."""
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            save_mappings(self.extractor, self._checkpoint_path(extension="zip"), self.num_timesteps)
        return result
