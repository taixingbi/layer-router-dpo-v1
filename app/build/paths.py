"""Filesystem paths for router gold data and built JSONL outputs."""

from __future__ import annotations

import os
from pathlib import Path

TRAIN_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TRAIN_REPO_ROOT / "data"
GOLDEN_TEST_ROOT = DATA_ROOT / "golden-test"
GOLDEN_DATA_DIR = GOLDEN_TEST_ROOT / "data"
EVAL_RESULT_ROOT = DATA_ROOT / "result"
GOLDEN_RESULT_DIR = EVAL_RESULT_ROOT
OUTPUT_ROOT = DATA_ROOT / "output"
DPO_OUTPUT_DIR = OUTPUT_ROOT / "dpo"
SFT_OUTPUT_DIR = OUTPUT_ROOT / "sft"


def router_model_slug(router_model: str) -> str:
    """Filesystem-safe slug from ROUTER_MODEL / --router-model."""
    return router_model.strip().replace("/", "__").replace(":", "__")


def golden_result_dir(router_model: str | None = None) -> Path:
    """Eval/DPO result dir under data/result/ (per-model subdir when ROUTER_MODEL set)."""
    model = (router_model or os.environ.get("ROUTER_MODEL", "")).strip()
    if model:
        return EVAL_RESULT_ROOT / router_model_slug(model)
    override = os.environ.get("RESULT_DIR", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (TRAIN_REPO_ROOT / p)
    return EVAL_RESULT_ROOT


def dataset_output_dir(method: str) -> Path:
    """Committed / built JSONL directory for dpo or sft."""
    m = method.strip().lower()
    if m == "dpo":
        return DPO_OUTPUT_DIR
    if m == "sft":
        return SFT_OUTPUT_DIR
    raise ValueError(f"unknown method: {method!r}; expected dpo or sft")


def dataset_subdir_relpath(method: str) -> str:
    """Repo-relative path for GitHub dataset fetch."""
    return dataset_output_dir(method).relative_to(TRAIN_REPO_ROOT).as_posix()


def orchestrator_root() -> Path:
    """Monorepo sibling or ORCHESTRATOR_ROOT env."""
    raw = os.environ.get("ORCHESTRATOR_ROOT", "").strip()
    if raw:
        return Path(raw).resolve()
    return (TRAIN_REPO_ROOT.parent / "layer-orchestrator-v1").resolve()
