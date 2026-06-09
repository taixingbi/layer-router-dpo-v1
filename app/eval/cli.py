"""CLI: `python -m app.eval` or `router-eval`."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from app.eval import golden_eval


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router-eval",
        description="Batch golden-test eval against orchestrator router.",
    )
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--result-dir", type=str, default="")
    parser.add_argument("--orchestrator-url", type=str, default="")
    parser.add_argument("--router-prompt-version", type=str, default="")
    parser.add_argument("--router-model", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--conversation-id", type=str, default="")
    parser.add_argument("--report-path", type=str, default="")
    args = parser.parse_args(argv)

    eval_argv: list[str] = []
    for flag, val in (
        ("--data-dir", args.data_dir),
        ("--result-dir", args.result_dir),
        ("--orchestrator-url", args.orchestrator_url),
        ("--router-prompt-version", args.router_prompt_version),
        ("--router-model", args.router_model),
        ("--conversation-id", args.conversation_id),
        ("--report-path", args.report_path),
    ):
        if val:
            eval_argv.extend([flag, val])
    if args.concurrency is not None:
        eval_argv.extend(["--concurrency", str(args.concurrency)])
    if args.timeout_s != 60.0:
        eval_argv.extend(["--timeout-s", str(args.timeout_s)])
    return golden_eval.main(eval_argv)


if __name__ == "__main__":
    raise SystemExit(main())
