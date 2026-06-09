"""Interpreter selection when train venv lacks orchestrator dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from app.build.paths import TRAIN_REPO_ROOT, orchestrator_root


def orchestrator_python() -> Path | None:
    """Prefer orchestrator repo venv, then train repo venv."""
    candidates = [
        orchestrator_root() / "venv" / "bin" / "python3",
        orchestrator_root() / ".venv" / "bin" / "python3",
        TRAIN_REPO_ROOT / ".venv" / "bin" / "python3",
        TRAIN_REPO_ROOT / "venv" / "bin" / "python3",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def orch_deps_available() -> bool:
    try:
        import pydantic  # noqa: F401
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        return False


def reexec_with_orch_python(argv: Sequence[str]) -> None:
    """Re-run this module/CLI with a Python that has orchestrator deps."""
    py = os.environ.get("ROUTER_EVAL_PYTHON")
    if py:
        path = Path(py)
    else:
        found = orchestrator_python()
        if found is None:
            print(
                "error: orchestrator dependencies missing (pydantic, langchain_core).\n"
                "  pip install -e ../layer-orchestrator-v1\n"
                "  or set ROUTER_EVAL_PYTHON to orchestrator venv python",
                file=sys.stderr,
            )
            raise SystemExit(1)
        path = found
    os.execv(str(path), [str(path), *argv])
