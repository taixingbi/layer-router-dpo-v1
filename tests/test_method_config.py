"""Tests for app/method_config.py."""

from __future__ import annotations

import pytest

from app.method_config import (
    DATASET_SUBDIR,
    base_model_to_slug,
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


def test_base_model_to_slug():
    assert base_model_to_slug("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b"
    assert base_model_to_slug("Qwen/Qwen2.5-7B-Instruct") == "qwen2.5-7b"


def test_default_hf_repo_suffix_from_base_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_REPO_MODEL", raising=False)
    monkeypatch.delenv("BASE_MODEL", raising=False)
    assert default_hf_repo_suffix("dpo") == "router-qwen2.5-1.5b-dpo-v1"
    assert default_hf_repo_suffix("sft") == "router-qwen2.5-1.5b-sft-v1"


def test_default_hf_repo_suffix_base_model_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_REPO_MODEL", raising=False)
    monkeypatch.setenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    assert default_hf_repo_suffix("dpo") == "router-qwen2.5-7b-dpo-v1"


def test_default_hf_repo_suffix_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_REPO_FEATURE", "router")
    monkeypatch.setenv("HF_REPO_MODEL", "qwen25-7b")
    monkeypatch.setenv("HF_REPO_VERSION", "v2")
    assert default_hf_repo_suffix("dpo") == "router-qwen25-7b-dpo-v2"
