#!/usr/bin/env python3
"""QLoRA DPO training for HuntAI intent router (16GB GPU safe defaults)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from load_jsonl import load_dpo_dataset  # noqa: E402


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA DPO train router from layer-orchestrator-v1 dpo-router/output/*.jsonl"
    )
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "2048")))
    p.add_argument("--num-train-epochs", type=float, default=float(os.getenv("NUM_TRAIN_EPOCHS", "2")))
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")),
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=int(os.getenv("GRAD_ACCUM", "8")))
    p.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "5e-5")))
    p.add_argument("--beta", type=float, default=float(os.getenv("DPO_BETA", "0.1")))
    p.add_argument("--lora-r", type=int, default=int(os.getenv("LORA_R", "32")))
    p.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "64")))
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-quant", action="store_true", help="Disable 4-bit (needs 24GB+ VRAM)")
    return p.parse_args(argv)


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )


def _lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_rows, val_rows = load_dpo_dataset(
        args.train_jsonl,
        args.val_jsonl,
        tokenizer=tokenizer,
    )
    if not train_rows:
        raise SystemExit(f"no training rows in {args.train_jsonl}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows) if val_rows else None

    model_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
    if not args.no_quant:
        model_kwargs["quantization_config"] = _bnb_config()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if not args.no_quant:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    peft_config = _lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=min(args.max_length, 1536),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if not args.no_quant else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")

    meta = {
        "base_model": args.base_model,
        "train_jsonl": str(args.train_jsonl.resolve()),
        "val_jsonl": str(args.val_jsonl.resolve()) if args.val_jsonl else None,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows) if val_rows else 0,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "quantized": not args.no_quant,
    }
    (args.output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved adapter -> {args.output_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
