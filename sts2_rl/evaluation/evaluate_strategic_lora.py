from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_examples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def normalized_decision(value: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    if not value:
        return None, None, None
    action = value.get("action")
    if not isinstance(action, dict):
        action = {}
    return value.get("status"), action.get("action"), action.get("index")


def valid_decision(value: dict[str, Any] | None, observation: dict[str, Any]) -> bool:
    status, action, index = normalized_decision(value)
    if status == "finished":
        return action == "none"
    if status != "act" or action not in observation.get("allowed_actions", []):
        return False
    if index is None:
        return True
    if not isinstance(index, int) or index < 0:
        return False
    candidates = observation.get("candidates", {})
    pools = {
        "choose_map_node": candidates.get("map_nodes", []),
        "choose_event_option": candidates.get("options", []),
        "choose_card": candidates.get("cards", []),
        "buy_shop_item": candidates.get("items", []),
    }
    pool = pools.get(action)
    if pool is None:
        return True
    return any(item.get("index") == index for item in pool if isinstance(item, dict))


def summarize(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    total = len(records)
    by_screen: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        bucket = by_screen[row["screen"]]
        bucket["count"] += 1
        for key in ("json_valid", "decision_valid", "status_match", "action_match", "exact_match"):
            bucket[key] += int(row[key])

    def rate(key: str) -> float:
        return round(sum(bool(row[key]) for row in records) / max(1, total), 4)

    screen_summary = {}
    for screen, counts in sorted(by_screen.items()):
        count = counts["count"]
        screen_summary[screen] = {"count": count}
        for key in ("json_valid", "decision_valid", "status_match", "action_match", "exact_match"):
            screen_summary[screen][f"{key}_rate"] = round(counts[key] / count, 4)
    return {
        "examples": total,
        "elapsed_seconds": round(elapsed, 2),
        "mean_latency_seconds": round(elapsed / max(1, total), 3),
        "json_valid_rate": rate("json_valid"),
        "decision_valid_rate": rate("decision_valid"),
        "status_match_rate": rate("status_match"),
        "action_match_rate": rate("action_match"),
        "exact_match_rate": rate("exact_match"),
        "by_screen": screen_summary,
    }


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], adapter_enabled: bool,
             max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    records = []
    started = time.perf_counter()
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.inference_mode():
        for number, row in enumerate(rows, 1):
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = parse_json(output)
            expected = parse_json(str(messages[-1]["content"]))
            observation = json.loads(messages[-2]["content"])
            predicted_tuple = normalized_decision(predicted)
            expected_tuple = normalized_decision(expected)
            records.append({
                "number": number,
                "screen": observation.get("screen", "unknown"),
                "expected": expected,
                "predicted": predicted,
                "raw_output": output,
                "json_valid": predicted is not None,
                "decision_valid": valid_decision(predicted, observation),
                "status_match": predicted_tuple[0] == expected_tuple[0],
                "action_match": predicted_tuple[:2] == expected_tuple[:2],
                "exact_match": predicted_tuple == expected_tuple,
            })
            if number == 1 or number % 25 == 0:
                print(f"[EVAL] adapter={adapter_enabled} {number}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    return records, summarize(records, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline comparison of base Qwen3 and strategic LoRA.")
    parser.add_argument("--test-file", default="logs/strategic_dataset_qwen_300/split/test.jsonl")
    parser.add_argument(
        "--adapter",
        default=str(PROJECT_ROOT / "artifacts" / "models" / "qwen3_4b_lora"),
    )
    parser.add_argument("--output-dir", default="logs/strategic_lora_evaluation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use env_llm.")
    adapter = Path(args.adapter)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_name = adapter_config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=quantization, device_map={"": 0}, dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    model.config.use_cache = True
    rows = load_examples(Path(args.test_file), args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"test_file": args.test_file, "adapter": str(adapter)}
    for name, enabled in (("base", False), ("lora", True)):
        records, summary = evaluate(model, tokenizer, rows, enabled, args.max_new_tokens)
        report[name] = summary
        result_path = output_dir / f"{name}_predictions.jsonl"
        result_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8",
        )
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps({name: summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
