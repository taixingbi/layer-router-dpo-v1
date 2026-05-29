"""CPU-only tests for scripts/load_jsonl.py (no GPU / no HF download)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = APP_ROOT / "scripts"
ORCH_DPO = APP_ROOT.parent / "layer-orchestrator-v1" / "dpo-router"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import load_jsonl as lj  # noqa: E402

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


class _MockTokenizer:
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "assistant: "
        return text


def test_records_to_dpo_rows():
    tok = _MockTokenizer()
    rows = lj.records_to_dpo_rows([SAMPLE_RECORD], tokenizer=tok)
    assert len(rows) == 1
    row = rows[0]
    assert row["chosen"] != row["rejected"]
    assert "greeting" in row["chosen"]
    assert "help" in row["rejected"]


def test_load_dpo_dataset_from_output(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(SAMPLE_RECORD) + "\n", encoding="utf-8")
    tok = _MockTokenizer()
    train_rows, val_rows = lj.load_dpo_dataset(train_path, None, tokenizer=tok)
    assert len(train_rows) == 1
    assert val_rows is None


@pytest.mark.skipif(
    not (ORCH_DPO / "output" / "train.jsonl").is_file(),
    reason="run layer-orchestrator-v1 dpo-router/run-build-dpo.sh first",
)
def test_load_production_train_jsonl():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    train_path = ORCH_DPO / "output" / "train.jsonl"
    val_path = ORCH_DPO / "output" / "val.jsonl"
    train_rows, val_rows = lj.load_dpo_dataset(train_path, val_path, tokenizer=tok)
    assert len(train_rows) >= 10
    assert val_rows is None or len(val_rows) >= 1
