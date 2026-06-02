"""CPU-only tests for app/load_jsonl.py (no GPU / no HF download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import load_jsonl as lj

APP_ROOT = Path(__file__).resolve().parents[1]
ORCH_DPO = APP_ROOT.parent / "layer-orchestrator-v1" / "router-eval" / "dpo-router"
ORCH_SFT = APP_ROOT.parent / "layer-orchestrator-v1" / "router-eval" / "sft-router"

SAMPLE_RECORD = {
    "prompt": [
        {"role": "system", "content": "You are the router."},
        {"role": "user", "content": "History:\n(none)\n\nLatest question:\nHi?"},
    ],
    "chosen": json.dumps(
        {
            "rewritten_question": "Hi?",
            "route": "greeting",
            "confidence": 0.95,
            "reason": "gold",
            "static_answer": "Hello",
            "repo": None,
        },
        separators=(",", ":"),
    ),
    "rejected": json.dumps(
        {
            "rewritten_question": "Hi?",
            "route": "help",
            "confidence": 0.95,
            "reason": "synthetic",
            "static_answer": None,
            "repo": None,
        },
        separators=(",", ":"),
    ),
}

SAMPLE_SFT_RECORD = {
    "messages": [
        {"role": "system", "content": "You are the router."},
        {"role": "user", "content": "History:\n(none)\n\nLatest question:\nHi?"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "rewritten_question": "Hi?",
                    "route": "greeting",
                    "confidence": 0.95,
                    "reason": "gold",
                    "static_answer": "Hello",
                    "repo": None,
                },
                separators=(",", ":"),
            ),
        },
    ],
}


class _MockTokenizer:
    """Minimal tokenizer stub with a chat template for unit tests."""

    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Render messages like a chat template without tokenizing."""
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "assistant: "
        return text


def test_records_to_dpo_rows():
    """Chosen and rejected completions differ while sharing the same prompt."""
    tok = _MockTokenizer()
    rows = lj.records_to_dpo_rows([SAMPLE_RECORD], tokenizer=tok)
    assert len(rows) == 1
    row = rows[0]
    assert row["chosen"] != row["rejected"]
    assert "greeting" in row["chosen"]
    assert "help" in row["rejected"]


def test_records_to_sft_rows():
    """SFT rows render full chat text including assistant route JSON."""
    tok = _MockTokenizer()
    rows = lj.records_to_sft_rows([SAMPLE_SFT_RECORD], tokenizer=tok)
    assert len(rows) == 1
    assert "greeting" in rows[0]["text"]
    assert "assistant:" in rows[0]["text"]


def test_load_dpo_dataset_from_output(tmp_path):
    """Load a single-record train file with no validation split."""
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(SAMPLE_RECORD) + "\n", encoding="utf-8")
    tok = _MockTokenizer()
    train_rows, val_rows = lj.load_dpo_dataset(train_path, None, tokenizer=tok)
    assert len(train_rows) == 1
    assert val_rows is None


def test_load_sft_dataset_from_output(tmp_path):
    """Load a single-record SFT train file."""
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(SAMPLE_SFT_RECORD) + "\n", encoding="utf-8")
    tok = _MockTokenizer()
    train_rows, val_rows = lj.load_sft_dataset(train_path, None, tokenizer=tok)
    assert len(train_rows) == 1
    assert val_rows is None


@pytest.mark.skipif(
    not (ORCH_DPO / "output" / "train.jsonl").is_file(),
    reason="run layer-orchestrator-v1 router-eval/dpo-router build first",
)
def test_load_production_dpo_jsonl():
    """Smoke-test real orchestrator DPO JSONL when the sibling repo is present."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    train_path = ORCH_DPO / "output" / "train.jsonl"
    val_path = ORCH_DPO / "output" / "val.jsonl"
    train_rows, val_rows = lj.load_dpo_dataset(train_path, val_path, tokenizer=tok)
    assert len(train_rows) >= 10
    assert val_rows is None or len(val_rows) >= 1


@pytest.mark.skipif(
    not (ORCH_SFT / "output" / "train.jsonl").is_file(),
    reason="run layer-orchestrator-v1 router-eval/sft-router build first",
)
def test_load_production_sft_jsonl():
    """Smoke-test real orchestrator SFT JSONL when the sibling repo is present."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    train_path = ORCH_SFT / "output" / "train.jsonl"
    val_path = ORCH_SFT / "output" / "val.jsonl"
    train_rows, val_rows = lj.load_sft_dataset(train_path, val_path, tokenizer=tok)
    assert len(train_rows) >= 10
    assert val_rows is None or len(val_rows) >= 1
