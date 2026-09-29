"""Logging-only combat outcome aggregation for PPO training."""
from stable_baselines3.common.callbacks import BaseCallback


class CombatStatsCallback(BaseCallback):
    """Accumulate combat results from vectorized environment info dictionaries."""

    def __init__(self, log_interval=1000):
        super().__init__()
        self.log_interval = int(log_interval)
        self.last_log_step = 0
        self.episodes = 0
        self.stats = {
            combat_type: {"wins": 0, "total": 0, "hp_lost": 0.0}
            for combat_type in ("monster", "elite", "boss")
        }

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info in infos:
            result = info.get("combat_result") if isinstance(info, dict) else None
            if not isinstance(result, dict) or result.get("type") not in self.stats:
                continue
            totals = self.stats[result["type"]]
            totals["total"] += 1
            totals["wins"] += int(bool(result.get("victory")))
            totals["hp_lost"] += float(result.get("hp_lost", 0.0) or 0.0)
        self.episodes += sum(bool(done) for done in dones)
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self._record_stats()
            self.last_log_step = self.num_timesteps
        return True

    def _record_stats(self):
        for combat_type, totals in self.stats.items():
            count = totals["total"]
            self.logger.record(f"combat/{combat_type}_win_rate",
                               totals["wins"] / count if count else 0.0)
            self.logger.record(f"combat/{combat_type}_avg_hp_lost",
                               totals["hp_lost"] / count if count else 0.0)
            self.logger.record(f"combat/{combat_type}_count", count)
        self.logger.record("combat/boss_reach_rate",
                           self.stats["boss"]["total"] / self.episodes if self.episodes else 0.0)
        self.logger.record("combat/episodes", self.episodes)


__all__ = ["CombatStatsCallback"]
