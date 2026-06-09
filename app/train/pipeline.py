"""End-to-end training pipeline: fetch dataset → validate load → train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.train.fetch_dataset import ensure_dataset
from app.train.load_jsonl import load_jsonl
from app.train.method_config import (
    METHODS,
    default_output_dir,
    local_data_dir,
    normalize_method,
    dataset_output_path,
)

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_method(args: argparse.Namespace) -> str:
    """Set args.method from CLI or env and return normalized value."""
    method = normalize_method(getattr(args, "method", None))
    args.method = method
    return method


def resolve_dataset_paths(args: argparse.Namespace) -> None:
    """Find or download train/val JSONL (sibling repo, then GitHub → data/{method}/)."""
    method = resolve_method(args)
    data_dir = local_data_dir(method)
    default_train = (data_dir / "train.jsonl").resolve()
    default_val = (data_dir / "val.jsonl").resolve()

    args.train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    if args.val_jsonl:
        args.val_jsonl = Path(args.val_jsonl).expanduser().resolve()

    for other in METHODS:
        if other == method:
            continue
        other_dir = local_data_dir(other).resolve()
        if other_dir in args.train_jsonl.parents:
            args.train_jsonl = default_train
            args.val_jsonl = default_val
            break

    if args.train_jsonl.is_file():
        return

    local_train = (dataset_output_path(method) / "train.jsonl").resolve()
    if local_train.is_file():
        print(f"using built dataset: {local_train.parent}", file=sys.stderr)
        args.train_jsonl = local_train
        local_val = (dataset_output_path(method) / "val.jsonl").resolve()
        args.val_jsonl = local_val if local_val.is_file() else None
        return

    if args.train_jsonl == default_train or data_dir.resolve() in args.train_jsonl.parents:
        print(f"fetch dataset from GitHub → {data_dir}/", file=sys.stderr)
        ensure_dataset(data_dir, method=method)
        args.train_jsonl = default_train
        args.val_jsonl = default_val if default_val.is_file() else None


def summarize_dataset(args: argparse.Namespace) -> None:
    """Print raw JSONL record counts before training."""
    train_n = len(load_jsonl(args.train_jsonl))
    val_n = 0
    if args.val_jsonl and args.val_jsonl.is_file():
        val_n = len(load_jsonl(args.val_jsonl))
    print(f"load: {train_n} train, {val_n} val records", file=sys.stderr)


def prepare_training(args: argparse.Namespace) -> None:
    """Fetch (if needed), resolve paths, and validate dataset is present."""
    resolve_dataset_paths(args)
    if not args.train_jsonl.is_file():
        raise SystemExit(
            f"missing {args.train_jsonl} — use default data/ paths or pass --train-jsonl"
        )
    if args.val_jsonl and not args.val_jsonl.is_file():
        args.val_jsonl = None
    summarize_dataset(args)


def run_training(args: argparse.Namespace) -> int:
    """Run the full fetch → load → train pipeline."""
    method = resolve_method(args)
    prepare_training(args)
    print(f"train: starting {method.upper()}", file=sys.stderr)
    if method == "sft":
        from app.train.train_sft import run as train_run
    else:
        from app.train.train_dpo import run as train_run
    return train_run(args)


__all__ = ["default_output_dir", "run_training", "prepare_training", "resolve_method"]
