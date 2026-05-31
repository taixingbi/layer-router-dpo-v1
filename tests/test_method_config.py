"""Tests for app/method_config.py."""

from __future__ import annotations

import pytest

from app.method_config import (
    DATASET_SUBDIR,
    dataset_subdir,
    default_hf_repo_suffix,
    local_data_dir,
    normalize_method,
)


def test_normalize_method_defaults_to_dpo():
    assert normalize_method(None) == "dpo"
    assert normalize_method("sft") == "sft"


def test_dataset_subdirs():
    assert DATASET_SUBDIR["dpo"] == "aval/dpo-router/output"
    assert DATASET_SUBDIR["sft"] == "aval/sft-router/output"
    assert dataset_subdir("dpo") == "aval/dpo-router/output"
    assert dataset_subdir("sft") == "aval/sft-router/output"


def test_local_data_dir_per_method():
    assert local_data_dir("dpo").name == "dpo"
    assert local_data_dir("sft").name == "sft"
    assert local_data_dir("dpo") != local_data_dir("sft")


def test_default_hf_repo_suffix():
    assert default_hf_repo_suffix("dpo") == "router-qwen25-1.5b-dpo-v1"
    assert default_hf_repo_suffix("sft") == "router-qwen25-1.5b-sft-v1"


def test_default_hf_repo_suffix_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_REPO_FEATURE", "router")
    monkeypatch.setenv("HF_REPO_MODEL", "qwen25-7b")
    monkeypatch.setenv("HF_REPO_VERSION", "v2")
    assert default_hf_repo_suffix("dpo") == "router-qwen25-7b-dpo-v2"
