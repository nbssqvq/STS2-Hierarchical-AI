# Slay the Spire 2 Hierarchical AI
[简体中文](README.zh-CN.md)

An experimental hierarchical agent for **Slay the Spire 2**. A MaskablePPO policy chooses combat actions, while a Qwen-compatible strategic agent handles routes, shops, events, rest sites, and reward choices. The game is controlled through the local HTTP API exposed by [STS2 MCP](https://github.com/Gennadiyev/STS2MCP).

This is a research prototype. PPO and the strategic model are integrated, but consistent full-run wins are not guaranteed.

## Project layout

```text
sts2_rl/
  runtime/       Environment, features, masks, MCP client, PPO adapter, game runner
  strategy/      Strategic policy, routing rules, Ollama client, LoRA server
  training/      PPO/LoRA training and dataset tools
  evaluation/    PPO and LoRA evaluation and run analysis
  config.py      Project-root and default model paths
tools/           Manual HTTP interaction utility
mods/            Deployable STS2 MCP and training helper mods
artifacts/models/Curated 800k PPO checkpoint and Qwen strategic LoRA inference bundle
logs/, models/   Local logs, datasets, model caches, and training checkpoints; ignored by Git
```
The modified MCP release DLL and manifest are retained under `mods/McpBuild`; the public checkout does not contain the root MCP source needed to rebuild that binary. The separate `PunchOffInstantFix` helper mod retains its source and release files.

## Current stage and included models

The hierarchical PPO + strategic-agent loop is integrated. The included PPO artifact is the latest available checkpoint at **800,902 environment steps**. Its mapping JSON is required alongside the ZIP file for correct feature encoding. The included Qwen3-4B LoRA package contains the adapter and tokenizer files needed for inference; it does not contain optimizer/trainer state, so it cannot resume LoRA fine-tuning by itself.

The Qwen base model is not included. Download `Qwen/Qwen3-4B` separately into the local Hugging Face cache (about 8 GB in this setup). The base model is published under Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the [official model card](https://huggingface.co/Qwen/Qwen3-4B).

The adapter weights use Git LFS. Install Git LFS before cloning and fetch LFS objects after cloning. Account storage and transfer quotas depend on the GitHub plan.

## Requirements

- Windows 10/11 and Python 3.11
- Slay the Spire 2 with the STS2 MCP mod enabled
- The included `PunchOffInstantFix` helper mod for the training setup used by this project
- CUDA-capable PyTorch for PPO GPU use and LoRA inference/training
- .NET 9 SDK only if rebuilding the helper mod

## Setup

Install Git LFS, clone the repository, create the Python environments, and install their dependencies. These environment folders are machine-local and ignored by Git; the commands below create them after cloning.

```powershell
git lfs install
git clone https://github.com/nbssqvq/STS2-Hierarchical-AI.git
cd STS2-Hierarchical-AI
git lfs pull

py -3.11 -m venv env1
.\env1\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

deactivate
py -3.11 -m venv env_llm
.\env_llm\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
```

`git clone` does not create Python environments automatically. Run `py -m venv` and `pip install` as above. Training scripts create their output folders when needed; the Qwen base-model cache must be downloaded separately.

### Install the game mods

The repository includes prebuilt mod DLLs and manifests. Copy them into the game's `mods` directory; this does not require rebuilding the MCP mod.

```powershell
$gameDir = "D:\Steam\steamapps\common\Slay the Spire 2"
$modsDir = Join-Path $gameDir "mods"
New-Item -ItemType Directory -Force $modsDir | Out-Null
Copy-Item .\mods\McpBuild\bin\Release\net9.0\STS2_MCP.dll $modsDir
Copy-Item .\mods\McpBuild\mod_manifest.json (Join-Path $modsDir "STS2_MCP.json")
Copy-Item .\mods\PunchOffInstantFix\bin\Release\net9.0\PunchOffInstantFix.dll $modsDir
Copy-Item .\mods\PunchOffInstantFix\PunchOffInstantFix.json $modsDir
```

Enable both mods in the game and start the client. STS2 MCP is based on [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP), with local changes in the distributed binary. Its copyright and MIT license are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

To rebuild the project-owned helper mod against your installation, set `GameData` to the directory containing `sts2.dll`, `GodotSharp.dll`, and `0Harmony.dll`:

```powershell
$gameData = Join-Path $gameDir "data_sts2_windows_x86_64"
dotnet build .\mods\PunchOffInstantFix\PunchOffInstantFix.csproj -c Release -p:GameData="$gameData"
```

## Strategic inference

To use the included LoRA server, download the base model into the cache used by the server. Run these commands from the repository root:

```powershell
$env:HF_HOME = Join-Path $PWD "models\hf_cache"
env_llm\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-4B')"
env_llm\Scripts\python.exe -m sts2_rl.strategy.lora_strategic_server --adapter artifacts/models/qwen3_4b_lora --host 127.0.0.1 --port 11435 --model-name sts2-qwen3-4b-lora
```

In the PowerShell window running the agent, point the client to the server:

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11435"
$env:STS2_STRATEGIC_MODEL = "sts2-qwen3-4b-lora"
```

The included `run_lora_server_watchdog.cmd` starts the same server from `env_llm`. An Ollama-compatible endpoint can be used instead by setting `STS2_STRATEGIC_URL` and `STS2_STRATEGIC_MODEL` to that service. To prevent silent fallback when the configured model is unavailable, set `STS2_STRATEGIC_REQUIRED=1` and `STS2_STRATEGIC_ALLOW_FALLBACK=0`.

## Train and evaluate

Run package entry points from the repository root, using `env1`:

```powershell
env1\Scripts\python.exe -m sts2_rl.training.train --total-timesteps 20000 --n-envs 1 --model-path logs/my-run/model --log-dir logs/my-run --save-freq 10000 --eval-episodes 2
env1\Scripts\python.exe -m sts2_rl.training.train --continue-from artifacts/models/ppo_800k/final_model.zip --total-timesteps 100000 --model-path logs/my-resume/model --log-dir logs/my-resume --save-freq 10000 --eval-episodes 2
env1\Scripts\python.exe -m sts2_rl.evaluation.evaluate_model --model-path artifacts/models/ppo_800k/final_model.zip --episodes 10
env1\Scripts\python.exe -m sts2_rl.runtime.run_single_game --model-path artifacts/models/ppo_800k/final_model.zip --episodes 1
```

The checkpoint ZIP and matching `.mappings.json` are a pair. Keep both together when copying a model or resuming training. For command options, run a module with `--help`, for example `python -m sts2_rl.training.training_watchdog --help`.

LoRA training and strategic-data tools are in `sts2_rl.training`; PPO and LoRA evaluation and dataset analysis are in `sts2_rl.evaluation`. Run these commands from the repository root so relative data and log paths resolve consistently. Collected datasets, full LoRA training checkpoints, TensorBoard logs, and the base-model cache remain local and are ignored by Git.

## License

This repository does not yet declare a project-wide license. The MIT notice applies to STS2 MCP, and Apache-2.0 applies to the Qwen3-4B base model. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
