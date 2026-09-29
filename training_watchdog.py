from __future__ import annotations

import argparse
import base64
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_GAME_EXE = Path(r"D:\Steam\steamapps\common\Slay the Spire 2\SlayTheSpire2.exe")
DEFAULT_MCP_URL = "http://127.0.0.1:15526/api/v1/singleplayer"
INSTANT_MARKER = Path(os.environ.get("APPDATA", str(Path.home()))) / "SlayTheSpire2" / "sts2_rl_instant_mode.json"
TERMINAL_SCREENS = {"game_over", "victory", "defeat", "lose", "victory_screen", "defeat_screen"}
MENU_SCREENS = {"menu", "main_menu", "character_select"}


@dataclass(frozen=True)
class GameProcess:
    pid: int
    create_time: float


def powershell_json(script: str) -> Any:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        check=True, capture_output=True, text=True, timeout=10,
    )
    output = result.stdout.strip()
    return json.loads(output) if output else None


def keep_system_awake(enabled: bool) -> None:
    """Prevent Windows idle sleep while the unattended supervisor is active."""
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_execution_state = kernel32.SetThreadExecutionState
    set_execution_state.argtypes = [ctypes.c_uint]
    set_execution_state.restype = ctypes.c_uint
    flags = 0x80000000 | (0x00000001 if enabled else 0)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if not set_execution_state(flags):
        raise ctypes.WinError(ctypes.get_last_error())


def read_state(url: str, timeout: float = 4.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        state = json.loads(response.read().decode("utf-8"))
    if not isinstance(state, dict):
        raise ValueError("MCP state response was not a JSON object")
    return state


def screen_name(state: dict[str, Any]) -> str:
    return str(state.get("screen") or state.get("state_type") or "").strip().lower()


def state_signature(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)


def is_active_run(state: dict[str, Any]) -> bool:
    screen = screen_name(state)
    return screen not in MENU_SCREENS | TERMINAL_SCREENS and int(
        (state.get("run") or {}).get("floor", 0) or 0
    ) > 0


def find_game_process(game_exe: Path) -> GameProcess | None:
    escaped_name = game_exe.name.replace("'", "''")
    escaped_path = str(game_exe.resolve()).replace("'", "''")
    script = (
        f"Get-Process -Name '{Path(escaped_name).stem}' -ErrorAction SilentlyContinue "
        "| Where-Object { $_.Path -and $_.Path.Equals("
        f"'{escaped_path}', [StringComparison]::OrdinalIgnoreCase) }} "
        "| Select-Object -First 1 Id,@{Name='StartUtc';Expression={$_.StartTime.ToUniversalTime().ToString('o')}} "
        "| ConvertTo-Json -Compress"
    )
    try:
        result = powershell_json(script)
        if not result:
            return None
        started = dt.datetime.fromisoformat(result["StartUtc"].replace("Z", "+00:00"))
        return GameProcess(int(result["Id"]), started.timestamp())
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def instant_marker_ready(pid: int, process_started: float) -> bool:
    try:
        marker = json.loads(INSTANT_MARKER.read_text(encoding="utf-8"))
        started_utc = dt.datetime.fromtimestamp(process_started, tz=dt.timezone.utc)
        marked_utc = dt.datetime.fromisoformat(str(marker["utc"]).replace("Z", "+00:00"))
        return (int(marker.get("pid", -1)) == pid and marker.get("mode") == "Instant"
                and marked_utc >= started_utc - dt.timedelta(seconds=5))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def install_staged_instant_mod(game_exe: Path) -> None:
    """Install staged mod files only while the game process is not running."""
    if find_game_process(game_exe) is not None:
        raise RuntimeError("Refusing to replace Instant Mode mod while the game is running")
    mod_dir = game_exe.parent / "mods"
    for filename in ("PunchOffInstantFix.dll", "PunchOffInstantFix.json"):
        staged = mod_dir / f"{filename}.pending"
        target = mod_dir / filename
        if staged.is_file():
            shutil.copy2(staged, target)
            staged.unlink()
            print(f"[WATCHDOG] installed staged mod file: {target.name}", flush=True)


def launch_game(game_exe: Path) -> GameProcess:
    if not game_exe.is_file():
        raise FileNotFoundError(f"Game executable not found: {game_exe}")
    install_staged_instant_mod(game_exe)
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([str(game_exe)], cwd=str(game_exe.parent), creationflags=flags)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process = find_game_process(game_exe)
        if process is not None:
            print(f"[WATCHDOG] game launched pid={process.pid}", flush=True)
            return process
        time.sleep(1)
    raise RuntimeError("Game process did not appear after launch")


def stop_game(game_exe: Path, grace_seconds: float = 8.0) -> None:
    process = find_game_process(game_exe)
    if process is None:
        return
    print(f"[WATCHDOG] stopping game pid={process.pid}", flush=True)
    subprocess.run(
        ["taskkill.exe", "/PID", str(process.pid), "/T"],
        capture_output=True, text=True, timeout=grace_seconds,
    )
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and find_game_process(game_exe) is not None:
        time.sleep(0.5)
    process = find_game_process(game_exe)
    if process is not None:
        result = subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and find_game_process(game_exe) is not None:
            time.sleep(0.25)
        if find_game_process(game_exe) is not None:
            detail = (result.stderr or result.stdout or "no taskkill details").strip()
            raise RuntimeError(f"Could not stop game pid={process.pid}: {detail}")


def wait_for_ready(game_exe: Path, mcp_url: str, timeout: float) -> tuple[GameProcess, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    launch_attempts = 0
    stable_key = None
    stable_count = 0
    while time.monotonic() < deadline:
        process = find_game_process(game_exe)
        if process is None:
            if launch_attempts >= 2:
                raise RuntimeError(
                    "Game exited before Instant Mode/MCP became ready after two launches; "
                    "stopping to avoid a restart loop. Check Godot and Windows crash logs."
                )
            process = launch_game(game_exe)
            launch_attempts += 1
        if instant_marker_ready(process.pid, process.create_time):
            try:
                state = read_state(mcp_url)
                # Instant Mode can initialize before the game UI and MCP state
                # have finished loading. Allow the client 15 seconds to load,
                # then require three matching observations (2 seconds apart).
                key = (screen_name(state), state.get("state_type"), state.get("menu_screen"))
                if key[0] and key[1] and time.time() - process.create_time >= 15.0:
                    if key == stable_key:
                        stable_count += 1
                    else:
                        stable_key = key
                        stable_count = 1
                    if stable_count >= 3:
                        print(f"[WATCHDOG] game ready pid={process.pid} instant_mode=Instant "
                              f"screen={key[0]} (loaded 15s+, MCP stable for 3 polls)", flush=True)
                        return process, state
                else:
                    stable_key = None
                    stable_count = 0
            except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
                stable_key = None
                stable_count = 0
        time.sleep(2)
    raise TimeoutError(
        "Game/MCP did not become ready with Instant Mode confirmed. "
        "Install the updated PunchOffInstantFix mod and check the game log for [STS2_RL_PREFLIGHT]."
    )


def ensure_game_ready(game_exe: Path, mcp_url: str, timeout: float) -> tuple[GameProcess, dict[str, Any]]:
    process = find_game_process(game_exe)
    if process is not None and instant_marker_ready(process.pid, process.create_time):
        try:
            return process, read_state(mcp_url)
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass
    if process is not None:
        # A running game with no current-process marker is an old build or did
        # not load the preference-enabling mod. Restart it once to apply it.
        stop_game(game_exe)
        time.sleep(2)
    # wait_for_ready owns launches and caps retries; starting here as well
    # caused one extra launch before its retry counter began.
    return wait_for_ready(game_exe, mcp_url, timeout)


def process_window_responding(pid: int) -> bool | None:
    """Check the game's top-level Windows window; None means no window was found."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_windows = user32.EnumWindows
    enum_windows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
    enum_windows.restype = wintypes.BOOL
    get_window_pid = user32.GetWindowThreadProcessId
    get_window_pid.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    get_window_pid.restype = wintypes.DWORD
    is_window_visible = user32.IsWindowVisible
    is_window_visible.argtypes = [wintypes.HWND]
    is_window_visible.restype = wintypes.BOOL
    send_message_timeout = user32.SendMessageTimeoutW
    send_message_timeout.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
    ]
    send_message_timeout.restype = wintypes.LPARAM
    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _param: int) -> int:
        owner = wintypes.DWORD()
        get_window_pid(hwnd, ctypes.byref(owner))
        if owner.value == pid and is_window_visible(hwnd):
            found.append(int(hwnd))
            return 0
        return 1

    callback = callback_type(visit)
    enum_windows(callback, 0)
    if not found:
        return None
    result = ctypes.c_size_t()
    smto_abort_if_hung = 0x0002
    response = send_message_timeout(
        wintypes.HWND(found[0]), 0, 0, 0, smto_abort_if_hung, 1200, ctypes.byref(result)
    )
    return bool(response)


def checkpoint_steps(path: Path) -> int:
    metrics_name = "interrupted_metrics.json" if path.stem == "interrupted_model" else "complete_metrics.json"
    metadata = path.parent / metrics_name
    if metadata.is_file():
        try:
            return int(json.loads(metadata.read_text(encoding="utf-8"))["num_timesteps"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot determine training step count for checkpoint: {path}")


def latest_checkpoint(directory: Path, fallback: Path) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for path in directory.glob("**/*.zip"):
        if path.stem not in {"interrupted_model", "final_model"} and not re.search(r"_\d+_steps$", path.stem):
            continue
        try:
            candidates.append((checkpoint_steps(path), path.stat().st_mtime, path))
        except (OSError, ValueError):
            continue
    if not candidates:
        return fallback
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def stop_worker(process: subprocess.Popen[Any], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    print("[WATCHDOG] asking training worker to save an interrupted checkpoint", flush=True)
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=grace_seconds)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def worker_main(args: argparse.Namespace) -> int:
    from train import train

    model = train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        model_path=args.model_path,
        log_dir=args.log_dir,
        continue_from=args.continue_from,
        save_freq=args.save_freq,
    )
    metadata = {
        "num_timesteps": int(model.num_timesteps),
        "metrics": {
            key: float(value) for key, value in model.logger.name_to_value.items()
            if key.startswith("train/")
        },
    }
    (Path(args.log_dir) / "complete_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0


def supervise(args: argparse.Namespace) -> int:
    checkpoint = Path(args.continue_from).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    current_steps = checkpoint_steps(checkpoint)
    target_steps = current_steps + args.total_timesteps
    base_log = Path(args.log_dir).resolve()
    base_log.mkdir(parents=True, exist_ok=True)
    game_exe = Path(args.game_exe).resolve()
    restarts = 0
    periodic_restarts = 0
    last_restart_step = -1

    if args.force_game_restart:
        print("[WATCHDOG] forcing a clean game restart before smoke test", flush=True)
        stop_game(game_exe)
        time.sleep(2)
        process, state = wait_for_ready(game_exe, args.mcp_url, args.startup_timeout)
    else:
        process, state = ensure_game_ready(game_exe, args.mcp_url, args.startup_timeout)

    while current_steps < target_steps:
        remaining = target_steps - current_steps
        segment_steps = min(remaining, args.restart_every)
        segment_target = current_steps + segment_steps
        attempt_dir = base_log / (
            f"segment_{current_steps}_{segment_target}_crash{restarts}_"
            f"{dt.datetime.now():%Y%m%d_%H%M%S}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        model_path = attempt_dir / "final_model"
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
            "--continue-from", str(checkpoint), "--total-timesteps", str(segment_steps),
            "--n-envs", str(args.n_envs), "--save-freq", str(args.save_freq),
            "--model-path", str(model_path), "--log-dir", str(attempt_dir),
        ]
        print(f"[WATCHDOG] training segment start step={current_steps} "
              f"segment_target={segment_target} final_target={target_steps} "
              f"checkpoint={checkpoint}", flush=True)
        with (attempt_dir / "worker.log").open("a", encoding="utf-8", errors="replace") as log:
            worker = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )

            def relay_worker_output() -> None:
                assert worker.stdout is not None
                for line in worker.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()

            relay = threading.Thread(target=relay_worker_output, daemon=True)
            relay.start()
            last_signature = None
            last_change = time.monotonic()
            last_mcp_success = time.monotonic()
            hung_window_since: float | None = None
            reason = ""
            try:
                while worker.poll() is None:
                    now = time.monotonic()
                    game = find_game_process(game_exe)
                    if game is None:
                        reason = "game_process_exited"
                        break
                    responsive = process_window_responding(game.pid)
                    if responsive is False:
                        hung_window_since = hung_window_since or now
                        if now - hung_window_since >= args.window_hang_seconds:
                            reason = "game_window_not_responding"
                            break
                    else:
                        hung_window_since = None

                    try:
                        state = read_state(args.mcp_url)
                        last_mcp_success = now
                        signature = state_signature(state)
                        if signature != last_signature:
                            last_signature = signature
                            last_change = now
                        elif now - last_change >= args.stall_seconds:
                            reason = f"MCP state unchanged for {args.stall_seconds:.0f}s ({screen_name(state)})"
                            break
                    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
                        if now - last_mcp_success >= args.mcp_failure_seconds:
                            reason = f"MCP state unavailable for {args.mcp_failure_seconds:.0f}s"
                            break
                    time.sleep(args.poll_seconds)
            finally:
                if worker.poll() is None:
                    stop_worker(worker, args.interrupt_grace_seconds)
                relay.join(timeout=10)

        return_code = worker.poll()
        candidate = latest_checkpoint(attempt_dir, checkpoint)
        if return_code == 0:
            complete = attempt_dir / "complete_metrics.json"
            if not complete.is_file():
                raise RuntimeError(f"Worker exited successfully without completion metrics: {attempt_dir}")
            current_steps = int(json.loads(complete.read_text(encoding="utf-8"))["num_timesteps"])
            checkpoint = attempt_dir / "final_model.zip"
            mappings = attempt_dir / "final_model.mappings.json"
            if not checkpoint.is_file() or not mappings.is_file():
                raise RuntimeError(f"Segment checkpoint or mappings missing: {attempt_dir}")
            if current_steps != segment_target:
                raise RuntimeError(
                    f"Segment ended at {current_steps}, expected {segment_target}; "
                    "check n_steps alignment before continuing."
                )
            print(f"[WATCHDOG_SEGMENT_COMPLETE] step={current_steps} "
                  f"model={checkpoint} mappings={mappings}", flush=True)
            if current_steps >= target_steps:
                print(f"[WATCHDOG] requested training complete at step={current_steps}", flush=True)
                return 0

            periodic_restarts += 1
            print(f"[WATCHDOG] periodic restart {periodic_restarts}: stopping game before next segment",
                  flush=True)
            stop_game(game_exe)
            time.sleep(2)
            launch_game(game_exe)
            process, state = wait_for_ready(game_exe, args.mcp_url, args.startup_timeout)
            print(f"[WATCHDOG] periodic restart ready pid={process.pid} "
                  f"screen={screen_name(state)}; resuming from step={current_steps}", flush=True)
            continue

        if not reason:
            reason = f"training worker exited with code {return_code}"
        print(f"[WATCHDOG_RECOVERY] reason={reason}", flush=True)
        stop_worker(worker, args.interrupt_grace_seconds)
        candidate = latest_checkpoint(attempt_dir, checkpoint)
        candidate_steps = checkpoint_steps(candidate)
        if candidate_steps > current_steps:
            checkpoint, current_steps = candidate, candidate_steps
        elif last_restart_step == current_steps:
            raise RuntimeError(
                f"Watchdog recovery made no checkpoint progress at step {current_steps}; "
                f"inspect {attempt_dir / 'worker.log'}"
            )
        last_restart_step = current_steps
        restarts += 1
        if restarts > args.max_restarts:
            raise RuntimeError(f"Exceeded max restarts ({args.max_restarts}); last checkpoint={checkpoint}")

        stop_game(game_exe)
        time.sleep(2)
        launch_game(game_exe)
        process, state = wait_for_ready(game_exe, args.mcp_url, args.startup_timeout)
        # Leave any saved run intact. STS2Env/GameRunner will choose Continue
        # when the restarted client presents Continue/Abandon, and only start a
        # fresh run if continuation is unavailable or rejected.

    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--continue-from", required=True)
        parser.add_argument("--total-timesteps", type=int, required=True)
        parser.add_argument("--n-envs", type=int, default=1)
        parser.add_argument("--save-freq", type=int, default=10_240)
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--log-dir", required=True)
        return worker_main(parser.parse_args(sys.argv[1:]))

    parser = argparse.ArgumentParser(
        description="Run STS2 PPO under a watchdog that confirms Instant Mode and recovers stalled game sessions."
    )
    parser.add_argument("--continue-from", required=True, help="PPO checkpoint zip to resume from.")
    parser.add_argument("--total-timesteps", type=int, required=True,
                        help="Additional steps to train, even across automatic restarts.")
    parser.add_argument("--log-dir", default="logs/supervised_training")
    parser.add_argument("--game-exe", default=str(DEFAULT_GAME_EXE))
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=10_240)
    parser.add_argument("--restart-every", type=int, default=10_240,
                        help="Save a complete segment, then restart the game and training worker.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stall-seconds", type=float, default=480.0,
                        help="Restart only after the complete MCP state has been unchanged this long.")
    parser.add_argument("--mcp-failure-seconds", type=float, default=90.0)
    parser.add_argument("--window-hang-seconds", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--force-game-restart", action="store_true",
                        help="Close and relaunch the game before training; useful for restart smoke tests.")
    parser.add_argument("--interrupt-grace-seconds", type=float, default=90.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    args = parser.parse_args()
    if args.total_timesteps <= 0 or args.n_envs <= 0 or args.save_freq <= 0 or args.restart_every <= 0:
        parser.error("total-timesteps, n-envs, save-freq, and restart-every must be positive")
    if args.restart_every % 1024 != 0:
        parser.error("restart-every must be a multiple of PPO n_steps=1024")
    try:
        keep_system_awake(True)
        try:
            return supervise(args)
        finally:
            keep_system_awake(False)
    except KeyboardInterrupt:
        print("[WATCHDOG] stopped by user", flush=True)
        return 130
    except Exception as exc:
        print(f"[WATCHDOG_FATAL] {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
