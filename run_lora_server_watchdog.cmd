@echo off
setlocal
cd /d "%~dp0"
if not exist "logs" mkdir "logs"

:restart
echo [%date% %time%] Starting strategic LoRA server.>> "logs\lora_server_watchdog.log"
"env_llm\Scripts\python.exe" -u -m sts2_rl.strategy.lora_strategic_server --host 127.0.0.1 --port 11435 --model-name sts2-qwen3-4b-lora --cache-implementation offloaded >> "logs\lora_server_watchdog.log" 2>&1
set "exit_code=%ERRORLEVEL%"
echo [%date% %time%] Strategic LoRA server exited with code %exit_code%; restarting in 5 seconds.>> "logs\lora_server_watchdog.log"
timeout /t 5 /nobreak >nul
goto restart
