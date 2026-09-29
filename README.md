# Slay the Spire 2 Hierarchical AI
[简体中文](README.zh-CN.md)

An experimental hierarchical agent for **Slay the Spire 2**. It combines a MaskablePPO tactical policy for combat with a Qwen-based strategic agent for route, shop, event, rest-site, and reward decisions. The game is controlled through the local HTTP API exposed by [STS2 MCP](https://github.com/Gennadiyev/STS2MCP).

This is a research prototype. The intended outcome is to let the tactical policy focus on combat while the strategic agent handles longer-horizon choices, then improve both from collected runs. These are goals, not a claim of reliable wins or a completed solver.

## Project overview

- **Combat:** MaskablePPO selects cards, targets, potions, and end-turn actions. The current observation schema is 4 (216 features), with a 128×128 MLP policy. Dynamic card, status, and entity mappings are stored beside checkpoints.
- **Strategy:** A Qwen 3 4B-compatible service handles meaningful non-combat decisions and combat-related selection screens. A deterministic fallback and a rule guard validate decisions.
- **Shared game flow:** Training and single-run evaluation use the same environment and `GameRunner` for automatic non-combat progression.
- **Learning workflow:** PPO training, checkpoint migration, run evaluation, strategic trajectory collection, dataset curation, and optional LoRA fine-tuning are provided as separate scripts.

## Current stage

The project has an integrated PPO + strategic-LLM loop and working PPO/LoRA training and evaluation workflows. The local PPO run has reached roughly 800k environment steps; the checkpoint, LoRA adapter, datasets, and run logs are local artifacts and are not included in this repository. The current work is still focused on evaluating decisions, improving data quality, and making long runs more reliable. Full-run completion and stable win rates are not yet guaranteed.

## Requirements

- Windows 10/11 (the scripts and examples are currently Windows-oriented)
- Slay the Spire 2 installed
- Python 3.11
- .NET 9 SDK to rebuild the included game mods
- STS2 MCP enabled in the game; the Python client expects `http://127.0.0.1:15526` by default
- An Ollama-compatible strategic inference endpoint. Defaults: `http://127.0.0.1:11434` and model `qwen3:4b`; the included LoRA server can be used instead when you have an adapter

## Setup

Clone the repository, create a Python environment, and install the PPO/runtime dependencies:

```powershell
git clone <repository-url>
cd <repository-directory>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install STS2 MCP and the local training helper mod into the game's `mods` directory. The repository includes their source and the current release DLLs. To rebuild them, set the game directory and build against the assemblies shipped with your own game installation:

```powershell
$env:STS2_GAME_DIR = "C:\path\to\Slay the Spire 2"
$gameData = Join-Path $env:STS2_GAME_DIR "data_sts2_windows_x86_64"
$mods = Join-Path $env:STS2_GAME_DIR "mods"
dotnet build .\mods\McpBuild\STS2_MCP.csproj -c Release -p:GameData="$gameData"
dotnet build .\mods\PunchOffInstantFix\PunchOffInstantFix.csproj -c Release -p:GameData="$gameData"
New-Item -ItemType Directory -Force $mods | Out-Null
Copy-Item .\mods\McpBuild\bin\Release\net9.0\STS2_MCP.dll $mods
Copy-Item .\mods\McpBuild\mod_manifest.json (Join-Path $mods "STS2_MCP.json")
Copy-Item .\mods\PunchOffInstantFix\bin\Release\net9.0\PunchOffInstantFix.dll $mods
Copy-Item .\mods\PunchOffInstantFix\PunchOffInstantFix.json $mods
```

Enable the mods in the game settings and start the game before running an agent. The MCP source is based on [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its copyright, MIT license, and the local changes made to it.

For strategic LoRA fine-tuning, use a separate environment so the training dependencies do not conflict with the game runtime environment:

```powershell
py -3.11 -m venv .venv-llm
.\.venv-llm\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
```

## Configure strategic inference

For a base Ollama model, make sure the service and model are available, then set the endpoint and model in the same PowerShell window used to run the agent:

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11434"
$env:STS2_STRATEGIC_MODEL = "qwen3:4b"
```

For the project's Ollama-compatible LoRA server, start it with an adapter directory you have prepared:

```powershell
python lora_strategic_server.py --adapter "<path-to-adapter>" --host 127.0.0.1 --port 11435 --model-name sts2-qwen3-4b-lora
```

Then configure the agent process to use that server:

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11435"
$env:STS2_STRATEGIC_MODEL = "sts2-qwen3-4b-lora"
```

The strategic client permits fallback by default. Set `STS2_STRATEGIC_REQUIRED=1` and `STS2_STRATEGIC_ALLOW_FALLBACK=0` when you need training to fail instead of using fallback if the configured model is unavailable.

## Train and evaluate PPO

With the game, MCP mod, and strategic service ready, start a fresh PPO run:

```powershell
python train.py --total-timesteps 20000 --n-envs 1 --model-path models/my-run/model --log-dir logs/my-run --save-freq 10000 --eval-episodes 2
```

Resume from a checkpoint and train for additional environment steps:

```powershell
python train.py --continue-from "models/<checkpoint>.zip" --total-timesteps 100000 --model-path models/my-resume/model --log-dir logs/my-resume --save-freq 10000 --eval-episodes 2
```

Evaluate a saved model or run one game:

```powershell
python evaluate_model.py --model-path "models/<model>.zip" --episodes 10
python run_single_game.py --model-path "models/<model>.zip" --episodes 1
```

`training_watchdog.py` can supervise longer runs and restart the game/training worker at configured intervals. Run `python training_watchdog.py --help` for its options. Model weights, training logs, mappings, and datasets are intentionally excluded from Git.

## Strategic data and LoRA workflow

The repository includes scripts for collecting run trajectories, analyzing and filtering decisions, splitting datasets, and training/evaluating a strategic LoRA adapter. The collected datasets and adapter are local training artifacts. Check each script's `--help` output for the required dataset and output paths:

```powershell
python run_single_game.py --help
python analyze_collection.py --help
python split_strategic_dataset.py --help
python train_strategic_lora.py --help
python evaluate_strategic_lora.py --help
```

## Licensing

This repository does not yet declare a project-wide license. The MIT license in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) applies to the upstream STS2 MCP code and does not license this project's own code.
