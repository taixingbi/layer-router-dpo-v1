#!/usr/bin/env python3
"""Merge QLoRA adapter into full weights for vLLM --model (no --enable-lora)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model directory.")
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--adapter-dir", type=Path, required=True, help="Path to saved adapter (train output)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = p.parse_args(argv)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"merged model -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
