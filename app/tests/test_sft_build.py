"""Router SFT dataset builder tests."""

import json
from pathlib import Path

import pytest

from app.build.build_sft import build_sft_dataset, _sft_record
from app.build.gold import (
    GoldRow,
    build_router_completion,
    load_router_system_prompt,
)
from app.build.orch import OrchestratorAPI

from app.build.paths import GOLDEN_DATA_DIR

GOLD_DATA = GOLDEN_DATA_DIR


@pytest.fixture(autouse=True)
def _reset_orch_api():
    pytest.importorskip("pydantic")
    pytest.importorskip("langchain_core")
    OrchestratorAPI.reset()
    yield
    OrchestratorAPI.reset()


def test_build_router_completion_rag_sft():
    row = GoldRow(
        question="What is Taixing Bi's visa status?",
        expected_route="rag_private_kb",
        source_file="t.csv",
    )
    out = build_router_completion(row)
    assert out["route"] == "rag_private_kb"
    assert "route_detail" not in out


def test_sft_record_messages():
    row = GoldRow(
        question="What is Taixing Bi's visa status?",
        expected_route="rag_private_kb",
        source_file="t.csv",
    )
    completion = build_router_completion(row)
    rec = _sft_record(
        system_prompt="sys",
        row=row,
        completion=completion,
        meta={"question": row.question, "expected_route": "rag_private_kb"},
    )
    assert len(rec["messages"]) == 3
    assert rec["messages"][0]["role"] == "system"
    assert rec["messages"][1]["role"] == "user"
    assert rec["messages"][2]["role"] == "assistant"
    assistant = json.loads(rec["messages"][2]["content"])
    assert assistant["route"] == "rag_private_kb"
    assert "chosen" not in rec
    assert "rejected" not in rec


def test_build_sft_from_gold_csvs():
    prompt = load_router_system_prompt("router-v2.00")
    train, val, stats = build_sft_dataset(
        gold_data_dir=GOLD_DATA,
        system_prompt=prompt,
        include_seed_faq=False,
        include_hack=False,
        router_prompt_version="router-v2.00",
        val_ratio=0.1,
    )
    assert stats["rows_total"] >= 20
    assert stats["examples_written"] >= 10
    assert len(train) + len(val) == stats["examples_written"]
    sample = train[0]
    assert "messages" in sample and "meta" in sample
    assert len(sample["messages"]) == 3
    json.loads(sample["messages"][2]["content"])
