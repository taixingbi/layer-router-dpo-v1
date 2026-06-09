"""CLI: `python -m app.build` or `router-build`."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from app.build import build_dpo, build_sft


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip() in ("1", "true", "yes")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="router-build",
        description="Build router DPO/SFT training JSONL from gold CSVs.",
    )
    sub = parser.add_subparsers(dest="method", required=True)
    for method, mod in ("dpo", build_dpo), ("sft", build_sft):
        p = sub.add_parser(method, help=f"Build {method.upper()} JSONL")
        p.add_argument("--gold-data-dir", type=str, default="")
        p.add_argument("--output-dir", type=str, default="")
        p.add_argument("--router-prompt-version", type=str, default="")
        p.add_argument("--val-ratio", type=float, default=None)
        p.add_argument("--include-seed-faq", action="store_true", default=None)
        p.add_argument("--include-hack", action="store_true", default=None)
        if method == "dpo":
            p.add_argument("--result-dir", type=str, default="")
            p.add_argument("--fetch-live", action="store_true", default=None)
            p.add_argument("--orchestrator-url", type=str, default="")
            p.add_argument("--fetch-timeout-s", type=float, default=60.0)

    args = parser.parse_args(argv)
    mod = build_dpo if args.method == "dpo" else build_sft
    build_argv: list[str] = []
    for flag, val in (
        ("--gold-data-dir", args.gold_data_dir),
        ("--output-dir", args.output_dir),
        ("--router-prompt-version", args.router_prompt_version),
    ):
        if val:
            build_argv.extend([flag, val])
    if args.val_ratio is not None:
        build_argv.extend(["--val-ratio", str(args.val_ratio)])
    include_faq = args.include_seed_faq if args.include_seed_faq is not None else _env_flag("INCLUDE_SEED_FAQ")
    include_hack = args.include_hack if args.include_hack is not None else _env_flag("INCLUDE_HACK")
    if include_faq:
        build_argv.append("--include-seed-faq")
    if include_hack:
        build_argv.append("--include-hack")
    if args.method == "dpo":
        if args.result_dir:
            build_argv.extend(["--result-dir", args.result_dir])
        fetch = args.fetch_live if args.fetch_live is not None else _env_flag("FETCH_LIVE")
        if fetch:
            build_argv.append("--fetch-live")
        orch = args.orchestrator_url or os.getenv("ORCHESTRATOR_URL", "")
        if orch:
            build_argv.extend(["--orchestrator-url", orch])
        if args.fetch_timeout_s != 60.0:
            build_argv.extend(["--fetch-timeout-s", str(args.fetch_timeout_s)])
    return mod.main(build_argv)


if __name__ == "__main__":
    raise SystemExit(main())
