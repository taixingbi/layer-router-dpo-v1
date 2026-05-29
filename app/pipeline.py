"""End-to-end training pipeline: fetch dataset → validate load → train."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from app.env import REPO_ROOT
from app.fetch_dataset import ensure_dataset
from app.load_jsonl import load_jsonl

_DEFAULT_DATA = REPO_ROOT / "data"
_ORCH_DPO = REPO_ROOT.parent / "layer-orchestrator-v1" / "dpo-router" / "output"
_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def default_output_dir() -> Path:
    """Return a timestamped checkpoint directory under checkpoints/."""
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return REPO_ROOT / "checkpoints" / f"router-dpo-qwen25-1.5b-{ts}"


def resolve_dataset_paths(args: argparse.Namespace) -> None:
    """Find or download train/val JSONL (sibling repo, then GitHub → data/)."""
    args.train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    if args.val_jsonl:
        args.val_jsonl = Path(args.val_jsonl).expanduser().resolve()

    if args.train_jsonl.is_file():
        return

    orch_train = (_ORCH_DPO / "train.jsonl").resolve()
    if orch_train.is_file():
        print(f"using orchestrator dataset: {orch_train.parent}", file=sys.stderr)
        args.train_jsonl = orch_train
        orch_val = (_ORCH_DPO / "val.jsonl").resolve()
        args.val_jsonl = orch_val if orch_val.is_file() else None
        return

    default_train = (_DEFAULT_DATA / "train.jsonl").resolve()
    default_data = _DEFAULT_DATA.resolve()
    if args.train_jsonl == default_train or default_data in args.train_jsonl.parents:
        print("fetch dataset from GitHub → data/", file=sys.stderr)
        ensure_dataset(_DEFAULT_DATA)
        args.train_jsonl = default_train
        val_path = (_DEFAULT_DATA / "val.jsonl").resolve()
        args.val_jsonl = val_path if val_path.is_file() else None


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
    from app.train_dpo import run as train_run

    prepare_training(args)
    print("train: starting DPO", file=sys.stderr)
    return train_run(args)
