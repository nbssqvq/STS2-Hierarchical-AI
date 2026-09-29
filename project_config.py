import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PPO_MODEL = PROJECT_ROOT / "models" / "schema2_212dim_128x128_500k1.zip"
DEFAULT_OLLAMA_URL = os.environ.get("STS2_STRATEGIC_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("STS2_STRATEGIC_MODEL", "qwen3:4b")


__all__ = ["DEFAULT_OLLAMA_MODEL", "DEFAULT_OLLAMA_URL", "DEFAULT_PPO_MODEL", "PROJECT_ROOT"]
