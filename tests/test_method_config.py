"""Tests for app/method_config.py."""

from __future__ import annotations

from app.method_config import (
    DATASET_SUBDIR,
    DEFAULT_HF_REPO_SUFFIX,
    dataset_subdir,
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
    assert DEFAULT_HF_REPO_SUFFIX["dpo"] == "layer-router-dpo-v1"
    assert DEFAULT_HF_REPO_SUFFIX["sft"] == "layer-router-sft-v1"
