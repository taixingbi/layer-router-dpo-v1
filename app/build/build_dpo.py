"""Build router DPO JSONL from golden-test CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.build.gold import (
    GoldRow,
    build_router_completion,
    build_user_message,
    canonical_expected_route,
    completion_json,
    iter_gold_rows,
    load_router_system_prompt,
    router_llm_eligible,
    val_split,
    write_jsonl,
)
from app.build.orch import get_orch
from app.build.paths import DPO_OUTPUT_DIR, GOLDEN_DATA_DIR, golden_result_dir


def _synthetic_rejected_row(row: GoldRow) -> GoldRow:
    er = canonical_expected_route(row)
    if er in ("rag_private_kb", "github_search", "web_search"):
        return GoldRow(
            question=row.question,
            expected_route="help",
            source_file=row.source_file,
            history=row.history,
        )
    if er in ("greeting", "identity", "help", "capabilities"):
        return GoldRow(
            question=row.question,
            expected_route="rag_private_kb",
            source_file=row.source_file,
            history=row.history,
        )
    if er == "reject":
        return GoldRow(
            question=row.question,
            expected_route="help",
            source_file=row.source_file,
            history=row.history,
        )
    return GoldRow(
        question=row.question,
        expected_route="rag_private_kb",
        source_file=row.source_file,
        history=row.history,
    )


def _dpo_record(
    *,
    system_prompt: str,
    row: GoldRow,
    chosen: Dict[str, Any],
    rejected: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(row)},
        ],
        "chosen": completion_json(chosen),
        "rejected": completion_json(rejected),
        "meta": meta,
    }


def _read_result_csv(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            q = (raw.get("question") or "").strip()
            if not q:
                continue
            out[q] = {
                "expected_route": (raw.get("expected_route") or "").strip(),
                "actual_route": (raw.get("actual_route") or "").strip(),
                "route_match": (raw.get("route_match") or "").strip(),
                "rewritten_question": (raw.get("rewritten_question") or "").strip(),
                "actual_answer": (raw.get("actual_answer") or "").strip(),
            }
    return out


def _rejected_from_result(row: GoldRow, result: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if result.get("route_match") == "true":
        return None
    orch = get_orch()
    actual = orch.route.normalize_gold_expected_route((result.get("actual_route") or "").strip())
    if not actual:
        return None
    fake = GoldRow(
        question=row.question,
        expected_route=actual,
        source_file=row.source_file,
        history=row.history,
    )
    completion = build_router_completion(fake, label="eval_actual")
    rw = (result.get("rewritten_question") or "").strip()
    if rw:
        completion["rewritten_question"] = rw
    ans = (result.get("actual_answer") or "").strip()
    if ans and actual in ("greeting", "identity", "help", "capabilities", "clarify"):
        completion["static_answer"] = ans
    return completion


def _fetch_eval_decision(
    *,
    orchestrator_url: str,
    question: str,
    expected_route: str,
    router_prompt_version: str,
    timeout_s: float,
) -> Optional[Dict[str, Any]]:
    orch = get_orch()
    url = f"{orchestrator_url.rstrip('/')}/v1/orchestrator/eval/router"
    body = json.dumps(
        {
            "question": question,
            "expected_route": canonical_expected_route(
                GoldRow(question=question, expected_route=expected_route, source_file="live")
            ),
            "router_prompt_version": router_prompt_version,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return None
    route = orch.route.normalize_legacy_route_to_canonical((decision.get("route") or "").strip())
    if route not in orch.route.CANONICAL_ROUTES:
        return None
    return {
        "rewritten_question": decision.get("rewritten_question") or question,
        "route": route,
        "confidence": float(decision.get("confidence") or 0.5),
        "reason": decision.get("reason") or "live eval",
        "static_answer": decision.get("static_answer"),
        "repo": decision.get("repo"),
    }


def build_dpo_dataset(
    *,
    gold_data_dir: Path,
    result_dir: Optional[Path],
    system_prompt: str,
    include_seed_faq: bool,
    include_hack: bool,
    fetch_live: bool,
    orchestrator_url: str,
    router_prompt_version: str,
    fetch_timeout_s: float,
    val_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "rows_total": 0,
        "pairs_written": 0,
        "skipped_ineligible": 0,
        "skipped_match": 0,
        "rejected_source": {"result_csv": 0, "live_eval": 0, "synthetic": 0},
        "by_expected_route": {},
    }

    result_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
    if result_dir and result_dir.is_dir():
        for rp in result_dir.glob("*.csv"):
            result_cache[rp.stem] = _read_result_csv(rp)

    for row in iter_gold_rows(
        gold_data_dir,
        include_seed_faq=include_seed_faq,
        include_hack=include_hack,
    ):
        stats["rows_total"] += 1
        canonical = canonical_expected_route(row)
        stats["by_expected_route"][canonical] = stats["by_expected_route"].get(canonical, 0) + 1

        if not router_llm_eligible(row):
            stats["skipped_ineligible"] += 1
            continue

        chosen = build_router_completion(row, label="gold")
        rejected: Optional[Dict[str, Any]] = None
        rejected_src = "synthetic"

        stem = Path(row.source_file).stem
        result_row = result_cache.get(stem, {}).get(row.question)
        if result_row:
            rejected = _rejected_from_result(row, result_row)
            if rejected is not None:
                rejected_src = "result_csv"

        if rejected is None and fetch_live and orchestrator_url:
            live = _fetch_eval_decision(
                orchestrator_url=orchestrator_url,
                question=row.question,
                expected_route=row.expected_route,
                router_prompt_version=router_prompt_version,
                timeout_s=fetch_timeout_s,
            )
            if live and live.get("route") != chosen.get("route"):
                rejected = live
                rejected_src = "live_eval"
            elif live and live.get("route") == chosen.get("route"):
                stats["skipped_match"] += 1
                continue

        if rejected is None:
            rejected = build_router_completion(_synthetic_rejected_row(row), label="synthetic_rejected")

        meta = {
            "question": row.question,
            "expected_route": canonical,
            "source_file": row.source_file,
            "rejected_source": rejected_src,
            "router_prompt_version": router_prompt_version,
        }
        record = _dpo_record(
            system_prompt=system_prompt,
            row=row,
            chosen=chosen,
            rejected=rejected,
            meta=meta,
        )
        stats["pairs_written"] += 1
        stats["rejected_source"][rejected_src] = stats["rejected_source"].get(rejected_src, 0) + 1
        if val_split(row.question, val_ratio):
            val.append(record)
        else:
            train.append(record)

    return train, val, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    from app.build.runtime import orch_deps_available, reexec_with_orch_python

    if not orch_deps_available():
        reexec_with_orch_python(["-m", "app.build", "dpo", *(argv or [])])

    parser = argparse.ArgumentParser(description="Build router DPO JSONL from golden-test CSVs.")
    parser.add_argument("--gold-data-dir", type=Path, default=GOLDEN_DATA_DIR)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="eval result CSVs (default: golden_result_dir from ROUTER_MODEL)",
    )
    parser.add_argument("--output-dir", type=Path, default=DPO_OUTPUT_DIR)
    parser.add_argument("--router-prompt-version", default="router-v2.00")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-seed-faq", action="store_true")
    parser.add_argument("--include-hack", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    parser.add_argument("--orchestrator-url", default="")
    parser.add_argument("--fetch-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)

    orch_url = (args.orchestrator_url or os.getenv("ORCHESTRATOR_URL") or "").strip()
    if args.fetch_live and not orch_url:
        print("error: --fetch-live requires --orchestrator-url or ORCHESTRATOR_URL", file=sys.stderr)
        return 1

    from app.build.fetch_gold import ensure_gold_data

    gold_data_dir = ensure_gold_data(args.gold_data_dir)

    system_prompt = load_router_system_prompt(args.router_prompt_version)
    if args.result_dir is None:
        args.result_dir = golden_result_dir(os.environ.get("ROUTER_MODEL"))
    result_dir = args.result_dir if args.result_dir.is_dir() else None

    train, val, stats = build_dpo_dataset(
        gold_data_dir=gold_data_dir,
        result_dir=result_dir,
        system_prompt=system_prompt,
        include_seed_faq=args.include_seed_faq,
        include_hack=args.include_hack,
        fetch_live=args.fetch_live,
        orchestrator_url=orch_url,
        router_prompt_version=args.router_prompt_version,
        fetch_timeout_s=args.fetch_timeout_s,
        val_ratio=args.val_ratio,
    )

    out_dir = args.output_dir
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "build-stats.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(val)} val   -> {val_path}")
    print(f"stats -> {stats_path}")
    return 0
