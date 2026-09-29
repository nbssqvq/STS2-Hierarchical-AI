from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")
    return value


class LoraInferenceEngine:
    def __init__(self, adapter: Path, model_name: str, max_new_tokens: int = 160,
                 cache_implementation: str = "offloaded") -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; start this service with env_llm.")
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = config["base_model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=quantization, device_map={"": 0},
            dtype=torch.bfloat16, local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter, local_files_only=True)
        self.model.eval()
        self.model.config.use_cache = True
        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.cache_implementation = cache_implementation
        self.lock = threading.Lock()

    def _acquire_generation_slot(self) -> None:
        queued_at = time.perf_counter()
        self.lock.acquire()
        queue_wait = time.perf_counter() - queued_at
        if queue_wait >= 0.5:
            print(f"[LORA_QUEUE] waited={queue_wait:.2f}s", flush=True)

    def _generate_text(self, messages: list[dict[str, Any]], requested_tokens: int | None = None) -> str:
        self._acquire_generation_slot()
        inputs = None
        output = None
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to(self.model.device)
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False,
                    max_new_tokens=min(int(requested_tokens or self.max_new_tokens), self.max_new_tokens),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    cache_implementation=self.cache_implementation,
                )
            return self.tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
        finally:
            del output
            del inputs
            self.torch.cuda.empty_cache()
            self.lock.release()

    def generate(self, messages: list[dict[str, Any]], requested_tokens: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        generated = self._generate_text(messages, requested_tokens)
        try:
            decision = extract_json(generated)
        except (json.JSONDecodeError, ValueError, TypeError) as first_error:
            print(f"[LORA_GENERATION_INVALID] attempt=1 error={type(first_error).__name__}: "
                  f"{first_error} output={generated!r}", flush=True)
            repair_messages = list(messages) + [
                {"role": "assistant", "content": generated},
                {"role": "user", "content": (
                    "上一条输出不是有效 JSON。只输出一个符合原始 schema 的 JSON 对象；"
                    "不要 Markdown、解释、前后缀或思考过程。"
                )},
            ]
            generated = self._generate_text(repair_messages, requested_tokens)
            try:
                decision = extract_json(generated)
            except (json.JSONDecodeError, ValueError, TypeError) as second_error:
                print(f"[LORA_GENERATION_INVALID] attempt=2 error={type(second_error).__name__}: "
                      f"{second_error} output={generated!r}", flush=True)
                raise
        elapsed = time.perf_counter() - started
        print(f"[LORA_GENERATION_OK] duration={elapsed:.2f}s", flush=True)
        return {
            "model": self.model_name,
            "message": {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
            "done": True,
            "total_duration": int(elapsed * 1_000_000_000),
        }


def make_handler(engine: LoraInferenceEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, value: dict[str, Any]) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                self._send(200, {"models": [{"name": engine.model_name}]})
            elif self.path == "/health":
                self._send(200, {"status": "ok", "model": engine.model_name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/chat":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                if request.get("model") != engine.model_name:
                    self._send(404, {"error": f"model not found: {request.get('model')}"})
                    return
                messages = request.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                options = request.get("options") if isinstance(request.get("options"), dict) else {}
                result = engine.generate(messages, options.get("num_predict"))
            except Exception as exc:
                print(f"[LORA_REQUEST_ERROR] {type(exc).__name__}: {exc}\n"
                      f"{traceback.format_exc()}", flush=True)
                try:
                    self._send(503 if "inference busy" in str(exc) else 500,
                               {"error": f"{type(exc).__name__}: {exc}"})
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    print("[LORA_CLIENT_DISCONNECTED] error response was not delivered", flush=True)
                return
            try:
                self._send(200, result)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                print("[LORA_CLIENT_DISCONNECTED] generation completed after client disconnected",
                      flush=True)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[LORA_HTTP] {self.address_string()} {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama-compatible STS2 strategic LoRA server.")
    parser.add_argument(
        "--adapter",
        default=str(PROJECT_ROOT / "artifacts" / "models" / "qwen3_4b_lora"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--model-name", default="sts2-qwen3-4b-lora")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--cache-implementation", choices=("dynamic", "offloaded"),
                        default="offloaded")
    args = parser.parse_args()

    engine = LoraInferenceEngine(
        Path(args.adapter), args.model_name, args.max_new_tokens, args.cache_implementation,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    server.daemon_threads = True
    print(f"[LORA_SERVER_READY] http://{args.host}:{args.port} model={args.model_name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
