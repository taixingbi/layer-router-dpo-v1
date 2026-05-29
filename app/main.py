#!/usr/bin/env python3
"""Unified CLI: end-to-end train (fetch → load → DPO) and merge adapter."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from app.env import REPO_ROOT, load_dotenv
from app.pipeline import _DEFAULT_MODEL, default_output_dir, run_training

_DEFAULT_MAX_LENGTH = 1024
_DEFAULT_GRAD_ACCUM = 8
_DEFAULT_EPOCHS = 2


def _env_path(name: str, default: Path) -> Path:
    """Resolve a path from env or default."""
    raw = os.getenv(name)
    return Path(raw) if raw else default


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    """Register the train subcommand parser."""
    from app.pipeline import _DEFAULT_DATA

    p = sub.add_parser(
        "train",
        help="End-to-end: fetch dataset (if needed), load JSONL, QLoRA DPO train",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=_env_path("TRAIN_JSONL", _DEFAULT_DATA / "train.jsonl"),
    )
    p.add_argument(
        "--val-jsonl",
        type=Path,
        default=_env_path("VAL_JSONL", _DEFAULT_DATA / "val.jsonl"),
    )
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", _DEFAULT_MODEL))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint root (default: checkpoints/router-dpo-qwen25-1.5b-<timestamp>/)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("MAX_LENGTH", str(_DEFAULT_MAX_LENGTH))),
    )
    p.add_argument(
        "--num-train-epochs",
        type=float,
        default=float(os.getenv("NUM_TRAIN_EPOCHS", str(_DEFAULT_EPOCHS))),
    )
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")),
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.getenv("GRAD_ACCUM", str(_DEFAULT_GRAD_ACCUM))),
    )
    p.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "5e-5")))
    p.add_argument("--beta", type=float, default=float(os.getenv("DPO_BETA", "0.1")))
    p.add_argument("--lora-r", type=int, default=int(os.getenv("LORA_R", "32")))
    p.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "64")))
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-quant", action="store_true", help="Disable 4-bit (needs 24GB+ VRAM)")
    p.set_defaults(_handler=_cmd_train)


def _add_merge_parser(sub: argparse._SubParsersAction) -> None:
    """Register the merge subcommand parser."""
    p = sub.add_parser("merge", help="Merge LoRA adapter into full weights")
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", _DEFAULT_MODEL))
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.set_defaults(_handler=_cmd_merge)


def _normalize_argv(argv: Optional[Sequence[str]]) -> list[str]:
    """Default to the train pipeline when no subcommand is given."""
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        return ["train"]
    if args[0] not in ("train", "merge"):
        return ["train", *args]
    return args


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI with train and merge subcommands."""
    parser = argparse.ArgumentParser(
        prog="layer-router-dpo",
        description="Router DPO: fetch → load → train (default), or merge adapter for deploy.",
    )
    sub = parser.add_subparsers(dest="command")
    _add_train_parser(sub)
    _add_merge_parser(sub)
    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    """Run fetch → load → train pipeline."""
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    print(f"output-dir: {args.output_dir}", file=sys.stderr)
    return run_training(args)


def _cmd_merge(args: argparse.Namespace) -> int:
    """Run the merge subcommand."""
    from app.export_merge import run as merge_run

    return merge_run(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for python -m app.main and the layer-router-dpo console script."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    handler: Callable[[argparse.Namespace], int] = args._handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
