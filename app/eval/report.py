"""Markdown report for golden-test eval results."""

from __future__ import annotations

import csv
import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _match_rate(true_count: int, false_count: int) -> str:
    denom = true_count + false_count
    return f"{100.0 * true_count / denom:.1f}%" if denom else "n/a"


def _esc_cell(s: Any, max_len: int = 180) -> str:
    if s is None:
        return ""
    t = str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return (t[:max_len] + "…") if len(t) > max_len else t


def generate_report(
    *,
    report_path: Path,
    result_dir: Path,
    eval_url: str,
    orchestrator_base: str,
    concurrency: int,
    router_prompt_version: str,
    router_model: str = "",
) -> list[tuple[str, dict[str, int]]]:
    """Write Markdown report; return per-file stats for terminal summary."""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per_file: list[tuple[str, dict[str, int]]] = []
    tot = {"rows": 0, "true": 0, "false": 0, "null": 0, "other": 0}
    bad_items: list[tuple[str, str, str, str, str, str]] = []

    for path in sorted(glob.glob(str(result_dir / "*.csv"))):
        name = os.path.basename(path)
        c = {"rows": 0, "true": 0, "false": 0, "null": 0, "other": 0}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            norm = [x.strip().lstrip("\ufeff") for x in header]
            want = ["question", "expected_route", "actual_route", "route_match"]
            if norm[:4] != want:
                continue
            has_rw = len(norm) >= 5 and norm[4] == "rewritten_question"
            has_aa = len(norm) >= 6 and norm[5] == "actual_answer"
            for row in reader:
                if len(row) < 4:
                    c["other"] += 1
                    tot["other"] += 1
                    continue
                c["rows"] += 1
                tot["rows"] += 1
                q, er, ar, rm = row[0], row[1], row[2], row[3]
                rw = row[4] if has_rw and len(row) > 4 else ""
                aa = row[5] if has_aa and len(row) > 5 else ""
                m = (rm or "").strip().lower()
                if m == "true":
                    c["true"] += 1
                    tot["true"] += 1
                elif m == "false":
                    c["false"] += 1
                    tot["false"] += 1
                    bad_items.append((name, q, er, ar, rw, aa))
                elif m == "null":
                    c["null"] += 1
                    tot["null"] += 1
                else:
                    c["other"] += 1
                    tot["other"] += 1
        per_file.append((name, c))

    lines = [
        "# Router eval report",
        "",
        f"- **Generated (UTC):** {generated_at}",
        f"- **Orchestrator base:** `{orchestrator_base}`",
        f"- **Eval endpoint:** `{eval_url}`",
        f"- **Concurrency:** {concurrency}",
        f"- **Router prompt version:** `{router_prompt_version}` (`app/prompts/{router_prompt_version}.txt`)",
    ]
    if router_model:
        lines.append(f"- **Router model:** `{router_model}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total rows | {tot['rows']} |",
            f"| `route_match` = true | {tot['true']} |",
            f"| `route_match` = false | {tot['false']} |",
            f"| `route_match` = null | {tot['null']} |",
            f"| Other / short rows | {tot['other']} |",
            f"| **Match rate** (true / (true+false)) | **{_match_rate(tot['true'], tot['false'])}** |",
            "",
            "## Per file",
            "",
            "| File | Rows | true | false | null | other | Match rate |",
            "|------|-----:|-----:|------:|-----:|------:|------------|",
        ]
    )
    for name, c in per_file:
        lines.append(
            f"| `{name}` | {c['rows']} | {c['true']} | {c['false']} | {c['null']} | {c['other']} | "
            f"{_match_rate(c['true'], c['false'])} |"
        )
    lines.extend(
        [
            "",
            "## Bad items (`route_match` = false)",
            "",
            "| Source file | expected_route | actual_route | question | rewritten_question | actual_answer |",
            "|-------------|----------------|--------------|----------|--------------------|---------------|",
        ]
    )
    if bad_items:
        for name, q, er, ar, rw, aa in sorted(bad_items, key=lambda x: (x[0], x[1] or "")):
            lines.append(
                f"| `{_esc_cell(name)}` | {_esc_cell(er)} | {_esc_cell(ar)} | {_esc_cell(q)} | "
                f"{_esc_cell(rw, max_len=220)} | {_esc_cell(aa, max_len=220)} |"
            )
    else:
        lines.append("| — | — | — | — | — | *none* |")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return per_file
