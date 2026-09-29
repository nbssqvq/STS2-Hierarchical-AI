from __future__ import annotations

import argparse
import csv
import ctypes
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    ("train/approx_kl", "approx_kl"),
    ("train/entropy_loss", "entropy_loss"),
    ("train/value_loss", "value_loss"),
    ("train/policy_gradient_loss", "policy_gradient_loss"),
    ("train/explained_variance", "explained_variance"),
    ("time/fps", "fps"),
)
COMBAT_WIN_RATE_KEYS = (
    ("combat/monster_win_rate", "monster win rate"),
    ("combat/elite_win_rate", "elite win rate"),
    ("combat/boss_win_rate", "boss win rate"),
)


def enable_vt_output() -> bool:
    """Enable ANSI cursor control in a Windows console, when available."""
    if os.name != "nt" or not sys.stdout.isatty():
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError):
        return False


def _segment_key(path: Path) -> tuple[float, float]:
    worker_log = path / "worker.log"
    progress = path / "progress.csv"
    timestamps = [path.stat().st_ctime]
    for candidate in (worker_log, progress):
        try:
            timestamps.append(candidate.stat().st_mtime)
        except OSError:
            pass
    return max(timestamps), path.stat().st_ctime


def latest_progress_file(log_dir: Path) -> tuple[Path | None, Path | None]:
    """Return the active/latest segment and its progress CSV, if one exists."""
    direct_progress = log_dir / "progress.csv"
    segments = [path for path in log_dir.glob("segment_*") if path.is_dir()]
    if segments:
        segment = max(segments, key=_segment_key)
        progress = segment / "progress.csv"
        return segment, progress if progress.is_file() else None
    if direct_progress.is_file():
        return log_dir, direct_progress
    return None, None


def read_latest_row(progress_path: Path | None) -> tuple[dict[str, str] | None, float | None]:
    if progress_path is None:
        return None, None
    try:
        modified = progress_path.stat().st_mtime
        with progress_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return None, modified
        return rows[-1], modified
    except (OSError, csv.Error, UnicodeError):
        return None, None


def rollout_duration_history(log_dir: Path, limit: int = 5, rollout_steps: int = 1024) -> list[float]:
    """Estimate actual seconds per rollout from elapsed-time deltas in CSV logs."""
    progress_files = list(log_dir.glob("segment_*/progress.csv"))
    direct_progress = log_dir / "progress.csv"
    if direct_progress.is_file():
        progress_files.append(direct_progress)
    progress_files.sort(key=lambda path: (path.stat().st_ctime, path.stat().st_mtime))

    durations: list[float] = []
    for progress_file in progress_files:
        try:
            with progress_file.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error, UnicodeError):
            continue
        for previous, current in zip(rows, rows[1:]):
            previous_elapsed = as_float(previous, "time/time_elapsed")
            current_elapsed = as_float(current, "time/time_elapsed")
            previous_step = as_float(previous, "time/total_timesteps")
            current_step = as_float(current, "time/total_timesteps")
            if None in (previous_elapsed, current_elapsed, previous_step, current_step):
                continue
            delta_elapsed = current_elapsed - previous_elapsed
            delta_steps = current_step - previous_step
            if delta_elapsed > 0 and delta_steps > 0:
                durations.append(delta_elapsed * rollout_steps / delta_steps)
    return durations[-limit:]


def as_float(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def format_value(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def progress_snapshot(
    log_dir: Path, start_step: int, target_step: int, rollout_steps: int = 1024,
) -> tuple[list[str], int, float | None, float, float | None]:
    segment, progress_file = latest_progress_file(log_dir)
    row, modified = read_latest_row(progress_file)
    observed_step = as_float(row, "time/total_timesteps")
    fps = as_float(row, "time/fps")
    age = None if modified is None else max(0.0, time.time() - modified)
    durations = rollout_duration_history(log_dir, rollout_steps=rollout_steps)
    avg_rollout_seconds = sum(durations) / len(durations) if durations else None
    if observed_step is None:
        current_step = start_step
    else:
        estimated = observed_step
        if avg_rollout_seconds and age is not None:
            # Advance smoothly within the current rollout using measured rollout
            # duration, rather than relying on a potentially noisy/low FPS value.
            estimated += min(rollout_steps, int(age / avg_rollout_seconds * rollout_steps))
        current_step = min(target_step, max(start_step, int(estimated)))

    metrics = [f"Log directory: {log_dir}"]
    metrics.append(f"Current segment: {segment.name if segment else 'waiting for first segment'}")
    rollout_metrics = sorted(
        ((key, value) for key, value in (row or {}).items() if key.startswith("rollout/")),
        key=lambda item: item[0],
    )
    metrics.append("Latest rollout metrics:")
    if rollout_metrics:
        metrics.extend(f"  {key:32} {value or '—'}" for key, value in rollout_metrics)
    else:
        metrics.append("  waiting for first rollout")
    metrics.append("Latest training metrics:")
    for key, label in METRIC_KEYS:
        metrics.append(f"{label:23} {format_value(as_float(row, key))}")
    metrics.append("Combat win rates:")
    for key, label in COMBAT_WIN_RATE_KEYS:
        rate = as_float(row, key)
        shown_rate = "—" if rate is None else f"{rate * 100:.1f}%"
        metrics.append(f"{label:23} {shown_rate}")
    if row is None:
        state = "waiting for first PPO rollout metrics"
    elif age is not None and age > 120:
        state = f"no new rollout metrics for {age:.0f}s; progress estimate may be stale"
    else:
        state = "reading latest rollout; step estimate advances using measured rollout duration"
    metrics.append(f"Status: {state}")
    eta_seconds = None
    if observed_step is not None and avg_rollout_seconds is not None:
        remaining_rollouts = max(0.0, (target_step - observed_step) / rollout_steps)
        eta_seconds = max(0.0, remaining_rollouts * avg_rollout_seconds - (age or 0.0))
        metrics.append(
            f"Measured rollout time: {avg_rollout_seconds:.1f}s average "
            f"from {len(durations)} recent interval(s)"
        )
    elif observed_step is not None:
        metrics.append("Measured rollout time: collecting at least two rollout samples")
    return metrics, current_step, fps, age if age is not None else 0.0, eta_seconds


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "collecting rollout timing"
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02}h {minutes:02}m"
    if hours:
        return f"{hours}h {minutes:02}m"
    return f"{minutes}m"


def make_progress_line(current: int, start: int, target: int, eta_seconds: float | None,
                       width: int = 36) -> str:
    total = max(1, target - start)
    completed = min(total, max(0, current - start))
    fraction = completed / total
    filled = min(width - 1, int(width * fraction))
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    return (f"[{bar}] {fraction * 100:6.2f}%  "
            f"{current:,}/{target:,} steps  ETA {format_eta(eta_seconds)}")


def render_tty(metrics: list[str], progress: str) -> None:
    # Redraw metrics at the top and pin the progress bar to the console's last row.
    rows = shutil.get_terminal_size((100, 24)).lines
    visible_metrics = metrics[:max(1, rows - 2)]
    sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(visible_metrics))
    sys.stdout.write(f"\x1b[{rows};1H{progress[:shutil.get_terminal_size((100, 24)).columns]}")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PPO training dashboard.")
    parser.add_argument("--log-dir", required=True, help="Watchdog log directory passed to training_watchdog.py.")
    parser.add_argument("--start-step", type=int, required=True, help="Checkpoint step at the start of this run.")
    parser.add_argument("--target-step", type=int, required=True, help="Absolute target step for this run.")
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    args = parser.parse_args()
    log_dir = Path(args.log_dir).resolve()
    if args.target_step <= args.start_step:
        parser.error("target-step must be greater than start-step")

    fixed_bottom = enable_vt_output()
    last_fallback_step = args.start_step
    last_metrics_signature: tuple[Any, ...] | None = None
    try:
        while True:
            metrics, current_step, fps, age, eta_seconds = progress_snapshot(
                log_dir, args.start_step, args.target_step,
            )
            progress = make_progress_line(current_step, args.start_step, args.target_step, eta_seconds)
            if fixed_bottom:
                render_tty(metrics, progress)
            else:
                signature = tuple(metrics)
                if signature != last_metrics_signature:
                    print("\n".join(metrics), flush=True)
                    last_metrics_signature = signature
                # Without cursor control, print one stable snapshot per ~100 estimated steps.
                if current_step - last_fallback_step >= 100 or current_step >= args.target_step:
                    print(progress, flush=True)
                    last_fallback_step = current_step
            if current_step >= args.target_step:
                break
            time.sleep(max(0.25, args.refresh_seconds))
    except KeyboardInterrupt:
        print("\n[monitor] stopped; training process was not changed.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
