"""Method-aware paths and names for DPO vs SFT router training."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from app.env import REPO_ROOT

METHODS = ("dpo", "sft")

DATASET_SUBDIR = {
    "dpo": "router-eval/dpo-router/output",
    "sft": "router-eval/sft-router/output",
}

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_HF_REPO_FEATURE = "router"
DEFAULT_HF_REPO_VERSION = "0.00"

_BASE_MODEL_SUFFIXES = ("-instruct", "-chat", "-base")


def base_model_to_slug(base_model: str) -> str:
    """Derive a Hub/checkpoint slug from a HuggingFace model id."""
    name = base_model.strip().rsplit("/", 1)[-1].lower()
    for suffix in _BASE_MODEL_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or "model"


DEFAULT_HF_REPO_MODEL = base_model_to_slug(DEFAULT_BASE_MODEL)

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


def default_output_dir(method: str, *, base_model: str | None = None) -> Path:
    """Timestamped checkpoint directory under checkpoints/."""
    method = normalize_method(method)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    slug = hf_repo_model(base_model=base_model)
    return REPO_ROOT / "checkpoints" / f"router-{method}-{slug}-{ts}"


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


def hf_repo_feature() -> str:
    """Hub repo name segment: feature (default router). Env: HF_REPO_FEATURE."""
    raw = os.getenv("HF_REPO_FEATURE", DEFAULT_HF_REPO_FEATURE).strip()
    return raw or DEFAULT_HF_REPO_FEATURE


def hf_repo_model(*, base_model: str | None = None) -> str:
    """Hub repo model segment: derived from BASE_MODEL unless HF_REPO_MODEL is set."""
    explicit = os.getenv("HF_REPO_MODEL", "").strip()
    if explicit:
        return explicit
    raw = (base_model or os.getenv("BASE_MODEL") or DEFAULT_BASE_MODEL).strip()
    return base_model_to_slug(raw) if raw else DEFAULT_HF_REPO_MODEL


def hf_repo_version() -> str:
    """Hub repo name segment: version tag (default 0.00). Env: HF_REPO_VERSION."""
    raw = os.getenv("HF_REPO_VERSION", DEFAULT_HF_REPO_VERSION).strip()
    return raw or DEFAULT_HF_REPO_VERSION


def default_hf_repo_suffix(method: str, *, base_model: str | None = None) -> str:
    """Default Hub repo suffix: {feature}-{model}-{method}-{version}."""
    method = normalize_method(method)
    return f"{hf_repo_feature()}-{hf_repo_model(base_model=base_model)}-{method}-{hf_repo_version()}"
