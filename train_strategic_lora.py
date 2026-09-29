from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_MODEL = "Qwen/Qwen3-4B"
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf_cache"))


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> list[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def render_example(example: Dict[str, Any], tokenizer: Any) -> Dict[str, str]:
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Each example must contain system/user and assistant messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError("The final message must be the assistant completion")
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    completion = str(messages[-1].get("content", ""))
    eos = tokenizer.eos_token or ""
    if eos and not completion.endswith(eos):
        completion += eos
    return {"prompt": prompt, "completion": completion}


def prepare_rows(rows: Iterable[Dict[str, Any]], tokenizer: Any) -> list[Dict[str, str]]:
    return [render_example(row, tokenizer) for row in rows]


def token_length_report(rows: list[Dict[str, str]], tokenizer: Any,
                        thresholds: tuple[int, ...] = (1024, 1536, 2048, 3072, 4096)) -> Dict[str, Any]:
    lengths = []
    prompt_lengths = []
    completion_lengths = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(len(completion_ids))
        lengths.append(len(prompt_ids) + len(completion_ids))
    ordered = sorted(lengths)

    def percentile(value: float) -> int:
        if not ordered:
            return 0
        index = min(len(ordered) - 1, max(0, math.ceil(value * len(ordered)) - 1))
        return ordered[index]

    return {
        "examples": len(lengths),
        "tokens": {
            "min": min(lengths, default=0), "mean": round(sum(lengths) / max(1, len(lengths)), 1),
            "p50": percentile(0.50), "p90": percentile(0.90), "p95": percentile(0.95),
            "p99": percentile(0.99), "max": max(lengths, default=0),
        },
        "prompt_mean": round(sum(prompt_lengths) / max(1, len(prompt_lengths)), 1),
        "completion_mean": round(sum(completion_lengths) / max(1, len(completion_lengths)), 1),
        "over_threshold": {str(value): sum(length > value for length in lengths) for value in thresholds},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the STS2 strategic Qwen3 LoRA adapter.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default="logs/strategic_dataset_qwen_300/split/train.jsonl")
    parser.add_argument("--validation-file", default="logs/strategic_dataset_qwen_300/split/validation.jsonl")
    parser.add_argument("--output-dir", default="models/strategic_qwen3_4b_lora")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const=True)
    parser.add_argument("--base-adapter", type=str,
                        help="Initialize a new fine-tuning run from an existing LoRA adapter.")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--analysis-output",
                        default="logs/strategic_dataset_qwen_300/token_length_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script with env_llm, not env1.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = prepare_rows(load_jsonl(args.train_file, args.sample_limit), tokenizer)
    validation_rows = prepare_rows(load_jsonl(args.validation_file, args.sample_limit), tokenizer)
    report = {
        "model": args.model,
        "train": token_length_report(train_rows, tokenizer),
        "validation": token_length_report(validation_rows, tokenizer),
        "configured_max_length": args.max_length,
    }
    analysis_path = Path(args.analysis_output)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.analyze_only:
        return

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization, device_map={"": 0}, dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False
    lora = None if args.base_adapter else LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    if args.base_adapter:
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True,
                                         local_files_only=True)
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        run_name="sts2-qwen3-4b-qlora",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
