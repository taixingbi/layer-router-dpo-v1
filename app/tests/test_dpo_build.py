"""Router DPO dataset builder tests."""

import json
from pathlib import Path

import pytest

from app.build.build_dpo import build_dpo_dataset
from app.build.gold import GoldRow, build_router_completion, load_router_system_prompt
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


def test_build_router_completion_rag():
    row = GoldRow(
        question="What is Taixing Bi's visa status?",
        expected_route="rag_private_kb",
        source_file="t.csv",
    )
    out = build_router_completion(row)
    assert out["route"] == "rag_private_kb"
    assert "route_detail" not in out
    assert "Taixing Bi" in out["rewritten_question"]


def test_build_dpo_from_gold_csvs():
    prompt = load_router_system_prompt("router-v2.00")
    train, val, stats = build_dpo_dataset(
        gold_data_dir=GOLD_DATA,
        result_dir=None,
        system_prompt=prompt,
        include_seed_faq=False,
        include_hack=False,
        fetch_live=False,
        orchestrator_url="",
        router_prompt_version="router-v2.00",
        fetch_timeout_s=1.0,
        val_ratio=0.1,
    )
    assert stats["rows_total"] >= 20
    assert stats["pairs_written"] >= 10
    assert len(train) + len(val) == stats["pairs_written"]
    sample = train[0]
    assert "prompt" in sample and "chosen" in sample and "rejected" in sample
    chosen = json.loads(sample["chosen"])
    rejected = json.loads(sample["rejected"])
    assert chosen["route"] != rejected["route"]
    assert "route_detail" not in chosen
