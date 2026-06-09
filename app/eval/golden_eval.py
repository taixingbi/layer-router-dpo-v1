"""Batch golden-test eval against POST /v1/orchestrator/eval/router."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.build.paths import GOLDEN_DATA_DIR, GOLDEN_RESULT_DIR, GOLDEN_TEST_ROOT
from app.eval.report import generate_report


@dataclass(frozen=True)
class EvalInputRow:
    question: str
    expected_route: str
    conversation_id: str
    history: List[Dict[str, Any]]


def _parse_eval_csv(path: Path) -> List[EvalInputRow]:
    rows: List[EvalInputRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            q = (raw.get("question") or "").strip()
            er = (raw.get("expected_route") or "").strip()
            if not (q and er):
                continue
            cid = (raw.get("conversation_id") or "").strip()
            hist_raw = (raw.get("history") or "").strip()
            if hist_raw:
                try:
                    hist = json.loads(hist_raw)
                    if not isinstance(hist, list):
                        hist = []
                except json.JSONDecodeError:
                    hist = []
            else:
                hist = []
            rows.append(
                EvalInputRow(
                    question=q,
                    expected_route=er,
                    conversation_id=cid,
                    history=hist,
                )
            )
    return rows


def _eval_row(
    *,
    eval_url: str,
    row: EvalInputRow,
    router_prompt_version: str,
    router_model: str,
    default_conv_id: str,
    suite_base: str,
    row_num: int,
    timeout_s: float,
) -> Dict[str, str]:
    conv_id = row.conversation_id or default_conv_id
    body: Dict[str, Any] = {
        "question": row.question,
        "expected_route": row.expected_route,
        "router_prompt_version": router_prompt_version,
        "conversation_id": conv_id,
        "history": row.history,
    }
    if router_model:
        body["router_model"] = router_model

    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"req-gold-{suite_base}-{row_num}",
        "X-Session-Id": f"ses-gold-{suite_base}",
        "X-Trace-Id": f"trc-gold-{suite_base}-{row_num}",
    }
    req = urllib.request.Request(
        eval_url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {
            "question": row.question,
            "expected_route": row.expected_route,
            "actual_route": "",
            "route_match": "",
            "rewritten_question": "",
            "actual_answer": "",
        }

    decision = payload.get("decision") or {}
    evaluation = payload.get("evaluation") or {}
    route_match = evaluation.get("route_match")
    if route_match is None:
        rm = "null"
    elif route_match:
        rm = "true"
    else:
        rm = "false"

    actual_route = (decision.get("route") or "") if isinstance(decision, dict) else ""
    rewritten = (decision.get("rewritten_question") or "") if isinstance(decision, dict) else ""
    actual_answer = ""
    if isinstance(decision, dict) and actual_route == "direct_reply":
        actual_answer = decision.get("answer") or ""

    return {
        "question": row.question,
        "expected_route": row.expected_route,
        "actual_route": actual_route,
        "route_match": rm,
        "rewritten_question": rewritten,
        "actual_answer": actual_answer,
    }


def _write_result_csv(out_path: Path, result_rows: List[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "expected_route",
        "actual_route",
        "route_match",
        "rewritten_question",
        "actual_answer",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)


def _process_csv(
    in_path: Path,
    *,
    eval_url: str,
    result_dir: Path,
    router_prompt_version: str,
    router_model: str,
    default_conv_id: str,
    concurrency: int,
    timeout_s: float,
    file_idx: int,
    file_total: int,
) -> None:
    base = in_path.stem
    out_path = result_dir / f"{base}.csv"
    inputs = _parse_eval_csv(in_path)
    if not inputs:
        print(f"[{file_idx}/{file_total}] {base} skip (empty)", file=sys.stderr)
        return

    conv_default = default_conv_id or f"conv-gold-{base}"

    def _job(item: tuple[int, EvalInputRow]) -> Dict[str, str]:
        i, row = item
        return _eval_row(
            eval_url=eval_url,
            row=row,
            router_prompt_version=router_prompt_version,
            router_model=router_model,
            default_conv_id=conv_default,
            suite_base=base,
            row_num=i,
            timeout_s=timeout_s,
        )

    results: List[Optional[Dict[str, str]]] = [None] * len(inputs)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_job, (i + 1, row)): i for i, row in enumerate(inputs)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    _write_result_csv(out_path, [r for r in results if r is not None])
    print(f"[{file_idx}/{file_total}] {base} {len(inputs)}/{len(inputs)}", file=sys.stderr)


def _discover_csvs(data_dir: Path) -> List[Path]:
    paths = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*/*.csv"))
    return paths


def _report_suffix(router_prompt_version: str, router_model: str) -> str:
    suffix = router_prompt_version
    if router_model:
        slug = router_model.replace("/", "__").replace(":", "__")
        suffix = f"{suffix}-{slug}"
    return suffix


def run_golden_eval(
    *,
    data_dir: Path,
    result_dir: Path,
    orchestrator_url: str,
    router_prompt_version: str,
    router_model: str = "",
    concurrency: int = 4,
    timeout_s: float = 60.0,
    default_conversation_id: str = "",
    report_path: Optional[Path] = None,
) -> int:
    orchestrator_url = orchestrator_url.rstrip("/")
    eval_url = f"{orchestrator_url}/v1/orchestrator/eval/router"
    inputs = _discover_csvs(data_dir)
    if not inputs:
        print(f"No CSV files found under {data_dir}", file=sys.stderr)
        return 1

    result_dir.mkdir(parents=True, exist_ok=True)
    if router_model:
        print(
            f"eval {router_prompt_version} · model {router_model} · {len(inputs)} files · {orchestrator_url}",
            file=sys.stderr,
        )
    else:
        print(f"eval {router_prompt_version} · {len(inputs)} files · {orchestrator_url}", file=sys.stderr)

    for idx, in_path in enumerate(inputs, start=1):
        _process_csv(
            in_path,
            eval_url=eval_url,
            result_dir=result_dir,
            router_prompt_version=router_prompt_version,
            router_model=router_model,
            default_conv_id=default_conversation_id,
            concurrency=concurrency,
            timeout_s=timeout_s,
            file_idx=idx,
            file_total=len(inputs),
        )

    if report_path is None:
        report_path = result_dir / f"router-eval-report-{_report_suffix(router_prompt_version, router_model)}.md"

    per_file = generate_report(
        report_path=report_path,
        result_dir=result_dir,
        eval_url=eval_url,
        orchestrator_base=orchestrator_url,
        concurrency=concurrency,
        router_prompt_version=router_prompt_version,
        router_model=router_model,
    )

    col_w = 48
    print(f"{'File':<{col_w}} Match rate")
    for name, c in per_file:
        denom = c["true"] + c["false"]
        rate = f"{100.0 * c['true'] / denom:.1f}%" if denom else "n/a"
        print(f"{name:<{col_w}} {rate}")
    tot_true = sum(c["true"] for _, c in per_file)
    tot_false = sum(c["false"] for _, c in per_file)
    denom = tot_true + tot_false
    rate = f"{100.0 * tot_true / denom:.1f}%" if denom else "n/a"
    print(f"{'(all suites)':<{col_w}} {rate}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch router golden-test eval.")
    parser.add_argument("--data-dir", type=Path, default=GOLDEN_DATA_DIR)
    parser.add_argument("--result-dir", type=Path, default=GOLDEN_RESULT_DIR)
    parser.add_argument(
        "--orchestrator-url",
        default=os.getenv("ORCHESTRATOR_URL", "http://192.168.86.179:30184"),
    )
    parser.add_argument("--router-prompt-version", default=os.getenv("ROUTER_PROMPT_VERSION", "router-v2.00"))
    parser.add_argument("--router-model", default=os.getenv("ROUTER_MODEL", ""))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "4")))
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--conversation-id", default=os.getenv("ROUTER_EVAL_CONVERSATION_ID", ""))
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args(argv)

    report_path = args.report_path
    if report_path is None:
        suffix = _report_suffix(args.router_prompt_version, args.router_model)
        report_path = args.result_dir / f"router-eval-report-{suffix}.md"

    return run_golden_eval(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        orchestrator_url=args.orchestrator_url,
        router_prompt_version=args.router_prompt_version,
        router_model=args.router_model,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        default_conversation_id=args.conversation_id,
        report_path=report_path,
    )
