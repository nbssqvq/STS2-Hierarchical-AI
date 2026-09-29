# 杀戮尖塔2分层智能体
[English](README.md)

这是一个用于**《杀戮尖塔2》**的分层智能体实验项目：战斗中由 MaskablePPO 负责出牌、选择目标、使用药水和结束回合；地图、商店、事件、篝火、奖励等战略决策由兼容 Qwen 的大模型负责。游戏状态和动作通过 [STS2 MCP](https://github.com/Gennadiyev/STS2MCP) 暴露的本地 HTTP API 传递。

本项目仍是研究原型。预期目标是让 PPO 专注战斗战术，让战略模型处理更长远的路线和资源选择，并利用采集数据持续改进；这属于计划目标，不代表已经能够稳定通关。

## 项目概述

- **战斗策略：** MaskablePPO 处理出牌、目标、药水和结束回合。当前特征 schema 为 4，共 216 维，策略网络为 128×128；动态卡牌、状态和实体映射与 checkpoint 配套保存。
- **战略策略：** 兼容 Qwen 3 4B 的服务处理非战斗决策及与当前战斗相关的选择界面，并通过规则保护和确定性 fallback 校验动作。
- **统一流程：** 训练与单局评估共用环境和 `GameRunner`，由同一套逻辑推进非战斗界面。
- **学习工具：** 项目提供 PPO 训练、断点迁移、评估、战略轨迹采集、数据筛选以及可选的 LoRA 微调脚本。

## 当前阶段

PPO 与战略大模型的协作链路、PPO/LoRA 训练和评估工作流已经集成。本地 PPO 训练约达到 80 万环境步；对应 checkpoint、LoRA adapter、数据集和日志不包含在仓库中。目前仍在评估决策质量、改善训练数据并提高长时间运行稳定性，尚不能保证稳定通关或达到固定胜率。

## 环境要求

- Windows 10/11（当前脚本和命令示例以 Windows 为主）
- 已安装《杀戮尖塔2》
- Python 3.11
- .NET 9 SDK（重新构建仓库内 Mod 时需要）
- 在游戏中启用 STS2 MCP；Python 客户端默认连接 `http://127.0.0.1:15526`
- 一个兼容 Ollama API 的战略推理服务。默认地址为 `http://127.0.0.1:11434`，默认模型为 `qwen3:4b`；有 LoRA adapter 时也可以使用仓库内的 LoRA 服务

## 安装

克隆仓库并安装 PPO 运行依赖：

```powershell
git clone <repository-url>
cd <repository-directory>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

将 STS2 MCP 和本项目的训练辅助 Mod 安装到游戏的 `mods` 目录。仓库保留了它们的源码和当前编译出的发布 DLL。要针对自己安装的游戏版本重新构建，请设置游戏目录并引用该游戏自带的程序集：

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

在游戏设置中启用 Mod 并启动游戏，然后再运行智能体。MCP 源码基于 [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP)，其版权、MIT 许可证和本地改动记录见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

训练战略 LoRA 时建议使用独立环境，以免大型训练依赖与游戏运行环境冲突：

```powershell
py -3.11 -m venv .venv-llm
.\.venv-llm\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
```

## 配置战略推理服务

使用基础 Ollama 模型时，先确保服务和模型可用，再在运行程序的同一个 PowerShell 窗口设置地址和模型：

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11434"
$env:STS2_STRATEGIC_MODEL = "qwen3:4b"
```

使用本项目的 Ollama 兼容 LoRA 服务时，将 adapter 路径替换为本地已有目录：

```powershell
python lora_strategic_server.py --adapter "<adapter目录>" --host 127.0.0.1 --port 11435 --model-name sts2-qwen3-4b-lora
```

然后让智能体连接该服务：

```powershell
$env:STS2_STRATEGIC_URL = "http://127.0.0.1:11435"
$env:STS2_STRATEGIC_MODEL = "sts2-qwen3-4b-lora"
```

默认情况下战略客户端允许 fallback。若续训时不允许服务故障后转为 fallback，请同时设置 `STS2_STRATEGIC_REQUIRED=1` 和 `STS2_STRATEGIC_ALLOW_FALLBACK=0`。

## PPO 训练与评估

准备好游戏、MCP Mod 和战略服务后，启动新的 PPO 训练：

```powershell
python train.py --total-timesteps 20000 --n-envs 1 --model-path models/my-run/model --log-dir logs/my-run --save-freq 10000 --eval-episodes 2
```

从 checkpoint 继续训练额外环境步：

```powershell
python train.py --continue-from "models/<checkpoint>.zip" --total-timesteps 100000 --model-path models/my-resume/model --log-dir logs/my-resume --save-freq 10000 --eval-episodes 2
```

评估模型或运行一局：

```powershell
python evaluate_model.py --model-path "models/<model>.zip" --episodes 10
python run_single_game.py --model-path "models/<model>.zip" --episodes 1
```

长时间训练可以由 `training_watchdog.py` 监督，并按配置重启游戏和训练工作进程，参数见 `python training_watchdog.py --help`。模型权重、训练日志、映射文件和数据集均由 Git 忽略。

## 战略数据与 LoRA

仓库包含游戏轨迹采集、决策分析与筛选、数据集拆分、战略 LoRA 训练和评估脚本。采集数据集和 adapter 属于本地训练产物，不提交到 Git。脚本参数和输入路径见：

```powershell
python run_single_game.py --help
python analyze_collection.py --help
python split_strategic_dataset.py --help
python train_strategic_lora.py --help
python evaluate_strategic_lora.py --help
```

## 许可证

本仓库尚未为项目自身选择统一许可证。[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中的 MIT 许可证适用于上游 STS2 MCP 代码，不代表本项目自身代码也采用 MIT 许可证。
