# 杀戮尖塔2分层智能体
[English](README.md)

这是一个用于《杀戮尖塔2》的分层智能体实验项目：战斗中由 MaskablePPO 选择出牌、目标、药水和结束回合动作；地图、商店、事件、篝火和奖励选择由兼容 Qwen 的战略智能体处理。游戏通过 [STS2 MCP](https://github.com/Gennadiyev/STS2MCP) 暴露的本地 HTTP API 控制。

本项目仍是研究原型。PPO 与战略智能体的协作链路已经集成，但尚不能保证稳定通关。

## 项目结构

```text
sts2_rl/
  runtime/       环境、特征、动作掩码、MCP 客户端、PPO 适配器、单局控制器
  strategy/      战略策略、路线规则、Ollama 客户端、LoRA 服务
  training/      PPO/LoRA 训练和数据集工具
  evaluation/    PPO/LoRA 评估及轨迹分析
  config.py      项目根目录和默认模型路径
tools/           手动 HTTP 交互工具
mods/            可安装到游戏中的 STS2 MCP 和训练辅助 Mod
artifacts/models/整理后的 800k PPO checkpoint 和 Qwen 战略 LoRA 推理包
logs/、models/   本机日志、数据集、模型缓存和训练断点；由 Git 忽略
```

项目根目录的 `.cs` 文件只作为本地参考源码，不提交到 Git。修改过的 MCP 发布 DLL 和 manifest 保存在 `mods/McpBuild`；公开仓库不包含重建该 DLL 所需的根目录 MCP 源码。独立的 `PunchOffInstantFix` 辅助 Mod 保留源码和发布文件。

## 当前阶段与模型文件

PPO 与战略智能体的协作流程已经集成。仓库包含目前最新的 **800,902 环境步** PPO checkpoint；正确读取特征还需要与 ZIP 同目录的映射 JSON。仓库也包含 Qwen3-4B LoRA 推理所需的 adapter 和 tokenizer 文件，但不含优化器和 Trainer 状态，因此该推理包本身不能用于恢复 LoRA 微调。

Qwen 基础模型不包含在仓库中。本地需要另行下载 `Qwen/Qwen3-4B` 到 Hugging Face 缓存；在当前环境约占 8 GB。基础模型采用 Apache-2.0 许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和[官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)。

LoRA 权重通过 Git LFS 管理。克隆前安装 Git LFS，克隆后执行 `git lfs pull`。GitHub 账户的 LFS 存储和流量额度取决于套餐。

## 环境要求

- Windows 10/11、Python 3.11
- 已安装《杀戮尖塔2》，并启用 STS2 MCP Mod
- 本项目训练配置使用的 `PunchOffInstantFix` 辅助 Mod
- PPO GPU 训练和 LoRA 推理/训练需要支持 CUDA 的 PyTorch
- 只有重新构建辅助 Mod 时才需要 .NET 9 SDK

## 安装

安装 Git LFS 后克隆仓库，创建 Python 虚拟环境并安装依赖。虚拟环境只属于本机，不上传到 Git；下面的命令会在克隆后创建它们。

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

`git clone` 不会自动创建 Python 环境；需要运行 `py -m venv` 和 `pip install`。训练脚本会在运行时创建输出目录；Qwen 基础模型缓存需要单独下载。

### 安装游戏 Mod

仓库内保留了预编译的 Mod DLL 和 manifest。将它们复制到游戏的 `mods` 目录即可，不需要重新构建 MCP Mod。

```powershell
$gameDir = "D:\Steam\steamapps\common\Slay the Spire 2"
$modsDir = Join-Path $gameDir "mods"
New-Item -ItemType Directory -Force $modsDir | Out-Null
Copy-Item .\mods\McpBuild\bin\Release\net9.0\STS2_MCP.dll $modsDir
Copy-Item .\mods\McpBuild\mod_manifest.json (Join-Path $modsDir "STS2_MCP.json")
Copy-Item .\mods\PunchOffInstantFix\bin\Release\net9.0\PunchOffInstantFix.dll $modsDir
Copy-Item .\mods\PunchOffInstantFix\PunchOffInstantFix.json $modsDir
```

在游戏中启用两个 Mod 并启动客户端。MCP 基于开源项目 [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)，发布 DLL 包含本地修改；版权和 MIT 许可证保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中。

如需针对本机游戏版本重新构建项目自有的辅助 Mod，将 `GameData` 指向含有 `sts2.dll`、`GodotSharp.dll` 和 `0Harmony.dll` 的目录：

```powershell
$gameData = Join-Path $gameDir "data_sts2_windows_x86_64"
dotnet build .\mods\PunchOffInstantFix\PunchOffInstantFix.csproj -c Release -p:GameData="$gameData"
```

## 战略推理

使用仓库中的 LoRA 服务前，先将基础模型下载到服务使用的本地缓存。在项目根目录运行：

```powershell
$env:HF_HOME = Join-Path $PWD "models\hf_cache"
env_llm\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-4B')"
env_llm\Scripts\python.exe -m sts2_rl.strategy.lora_strategic_server --adapter artifacts/models/qwen3_4b_lora --host 127.0.0.1 --port 11435 --model-name sts2-qwen3-4b-lora
```

在运行智能体的 PowerShell 窗口中指定服务：

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11435"
$env:STS2_STRATEGIC_MODEL = "sts2-qwen3-4b-lora"
```

`run_lora_server_watchdog.cmd` 也会从 `env_llm` 启动相同服务。也可以改用 Ollama 兼容服务并设置 `STS2_STRATEGIC_URL`、`STS2_STRATEGIC_MODEL`。如果服务不可用时不允许静默切换到 fallback，同时设置 `STS2_STRATEGIC_REQUIRED=1` 和 `STS2_STRATEGIC_ALLOW_FALLBACK=0`。

## 训练与评估

在项目根目录使用 `env1` 运行包入口：

```powershell
env1\Scripts\python.exe -m sts2_rl.training.train --total-timesteps 20000 --n-envs 1 --model-path logs/my-run/model --log-dir logs/my-run --save-freq 10000 --eval-episodes 2
env1\Scripts\python.exe -m sts2_rl.training.train --continue-from artifacts/models/ppo_800k/final_model.zip --total-timesteps 100000 --model-path logs/my-resume/model --log-dir logs/my-resume --save-freq 10000 --eval-episodes 2
env1\Scripts\python.exe -m sts2_rl.evaluation.evaluate_model --model-path artifacts/models/ppo_800k/final_model.zip --episodes 10
env1\Scripts\python.exe -m sts2_rl.runtime.run_single_game --model-path artifacts/models/ppo_800k/final_model.zip --episodes 1
```

模型 ZIP 和对应的 `.mappings.json` 必须配套保留。查看训练 watchdog 等程序的参数，可运行 `python -m sts2_rl.training.training_watchdog --help`。

LoRA 训练和战略数据工具位于 `sts2_rl.training`；PPO/LoRA 评估和轨迹分析位于 `sts2_rl.evaluation`。请从项目根目录运行这些模块，以保证相对数据路径和日志路径一致。采集数据集、完整 LoRA 训练断点、TensorBoard 日志和基础模型缓存均保留在本机，不提交到 Git。

## 许可证

本仓库尚未为项目自身指定统一许可证。MIT 许可证适用于 STS2 MCP；Apache-2.0 适用于 Qwen3-4B 基础模型。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
