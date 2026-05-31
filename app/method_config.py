"""Method-aware paths and names for DPO vs SFT router training."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from app.env import REPO_ROOT

METHODS = ("dpo", "sft")

DATASET_SUBDIR = {
    "dpo": "aval/dpo-router/output",
    "sft": "aval/sft-router/output",
}

DEFAULT_HF_REPO_SUFFIX = {
    "dpo": "layer-router-dpo-v1",
    "sft": "layer-router-sft-v1",
}

_ORCH_ROOT = REPO_ROOT.parent / "layer-orchestrator-v1"


def normalize_method(raw: str | None) -> str:
    """Return a validated training method (default dpo)."""
    method = (raw or os.getenv("TRAIN_METHOD") or "dpo").strip().lower()
    if method not in METHODS:
        raise SystemExit(f"invalid TRAIN_METHOD={method!r}; expected one of {METHODS}")
    return method


def local_data_dir(method: str) -> Path:
    """Per-method cache under data/ to avoid cross-method overwrite."""
    return REPO_ROOT / "data" / normalize_method(method)


def orch_sibling_path(method: str) -> Path:
    """Monorepo sibling path for orchestrator JSONL output."""
    method = normalize_method(method)
    return _ORCH_ROOT / DATASET_SUBDIR[method]


def default_output_dir(method: str) -> Path:
    """Timestamped checkpoint directory under checkpoints/."""
    method = normalize_method(method)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return REPO_ROOT / "checkpoints" / f"router-{method}-qwen25-1.5b-{ts}"


def dataset_subdir(method: str) -> str:
    """GitHub subdir for orchestrator dataset fetch."""
    method = normalize_method(method)
    override = os.getenv("ORCHESTRATOR_DATASET_SUBDIR")
    if override:
        return override.strip("/")
    legacy = os.getenv("ORCHESTRATOR_DPO_SUBDIR")
    if legacy and method == "dpo":
        return legacy.strip("/")
    return DATASET_SUBDIR[method]


def default_hf_repo_suffix(method: str) -> str:
    """Default Hugging Face repo name suffix for a training method."""
    return DEFAULT_HF_REPO_SUFFIX[normalize_method(method)]
