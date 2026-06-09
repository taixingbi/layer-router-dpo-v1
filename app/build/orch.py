"""Load layer-orchestrator-v1 `app` package (isolated from this repo's `app.train` / `app.build`)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

from app.build.paths import orchestrator_root


@contextmanager
def orchestrator_session() -> Iterator[Path]:
    """Prioritize orchestrator on sys.path; restore prior `app` modules on exit."""
    from pathlib import Path

    root = orchestrator_root()
    root_str = str(root)
    saved_path = list(sys.path)
    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "app" or k.startswith("app.")
    }
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    try:
        yield root
    finally:
        for k in list(sys.modules):
            if k == "app" or k.startswith("app."):
                sys.modules.pop(k, None)
        sys.path[:] = saved_path
        sys.modules.update(saved_modules)


class OrchestratorAPI:
    """Cached orchestrator modules (references stay valid after session ends)."""

    _instance: OrchestratorAPI | None = None

    def __init__(self) -> None:
        self.route: ModuleType
        self.rewrite: ModuleType
        self.github_route: ModuleType
        self.intent_router: ModuleType
        with orchestrator_session():
            from app.core import github_route, intent_router, rewrite
            from app.schemas import route

            self.route = route
            self.rewrite = rewrite
            self.github_route = github_route
            self.intent_router = intent_router

    @classmethod
    def get(cls) -> OrchestratorAPI:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear cached API (tests)."""
        cls._instance = None


def get_orch() -> OrchestratorAPI:
    return OrchestratorAPI.get()


def ensure_orchestrator_on_path() -> Path:
    """Legacy helper: enter orchestrator session (loads API via get_orch)."""
    from pathlib import Path

    root = orchestrator_root()
    get_orch()
    return root
