#!/usr/bin/env python3
"""Merge QLoRA adapter into full weights for vLLM --model (no --enable-lora)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from app.train.env import load_dotenv


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI flags and env-backed defaults for adapter merge."""
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model directory.")
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    p.add_argument("--adapter-dir", type=Path, required=True, help="Path to saved adapter (train output)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return p.parse_args(argv)


def _resolve_base_model(adapter_dir: Path, cli_base: str) -> str:
    """Prefer train_meta.json base_model when CLI still uses the default."""
    meta_path = adapter_dir.parent / "train_meta.json"
    if not meta_path.is_file():
        return cli_base
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return cli_base
    recorded = meta.get("base_model")
    if isinstance(recorded, str) and recorded:
        return recorded
    return cli_base


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Merge a trained LoRA adapter into full base weights and save for vLLM."""
    load_dotenv()
    return run(_parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Merge adapter using parsed CLI/env arguments."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = _resolve_base_model(args.adapter_dir, args.base_model)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
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
