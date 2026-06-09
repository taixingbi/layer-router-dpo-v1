"""Download router golden-test CSVs from layer-orchestrator-v1 GitHub."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.build.paths import GOLDEN_DATA_DIR, orchestrator_root

DEFAULT_ORCH_REPO = "taixingbi/layer-orchestrator-v1"
DEFAULT_REF = "main"
DEFAULT_GOLD_SUBDIR = "router-eval/golden-test/data"
_USER_AGENT = "layer-router-train-v1"

# Known suites under tools/ and internal/ on orchestrator main.
_GOLD_FILES: dict[str, tuple[str, ...]] = {
    "tools": (
        "router_github.csv",
        "router_rag_private_kb.csv",
        "router_web_search.csv",
    ),
    "internal": (
        "router_capabilities.csv",
        "router_greeting.csv",
        "router_help.csv",
        "router_identity.csv",
        "router_reject.csv",
    ),
}


def _has_gold_csvs(data_dir: Path) -> bool:
    return any(data_dir.glob("**/*.csv"))


def _raw_base_url(repo: str, ref: str, subdir: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{subdir.strip('/')}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            dest.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"fetch failed {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"fetch failed {url}: {exc.reason}") from exc


def _copy_local_gold(src: Path, dest: Path) -> None:
    for sub, names in _GOLD_FILES.items():
        for name in names:
            shutil.copy2(src / sub / name, dest / sub / name)


def download_gold_data(
    dest: Path | None = None,
    *,
    repo: str | None = None,
    ref: str | None = None,
    subdir: str | None = None,
) -> Path:
    """Fetch gold CSVs into dest (default data/golden-test/data)."""
    dest = (dest or GOLDEN_DATA_DIR).resolve()
    repo = repo or os.environ.get("ORCHESTRATOR_GOLD_REPO", DEFAULT_ORCH_REPO)
    ref = ref or os.environ.get("ORCHESTRATOR_REF", DEFAULT_REF)
    subdir = subdir or os.environ.get("ORCHESTRATOR_GOLD_SUBDIR", DEFAULT_GOLD_SUBDIR)
    base = _raw_base_url(repo, ref, subdir)

    jobs: list[tuple[str, Path, str]] = []
    for sub, names in _GOLD_FILES.items():
        for name in names:
            jobs.append((f"{base}/{sub}/{name}", dest / sub / name, f"{sub}/{name}"))

    for _, _, label in jobs:
        print(f"fetch gold {label} ({repo}@{ref})", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        list(pool.map(lambda job: _download(job[0], job[1]), jobs))
    print(f"gold data -> {dest}", file=sys.stderr)
    return dest


def ensure_gold_data(dest: Path | None = None) -> Path:
    """Use local gold CSVs, sibling orchestrator checkout, or GitHub download."""
    dest = (dest or GOLDEN_DATA_DIR).resolve()
    if _has_gold_csvs(dest):
        return dest

    sibling = orchestrator_root() / "router-eval/golden-test/data"
    if sibling.is_dir() and _has_gold_csvs(sibling):
        print(f"copy gold data from {sibling}", file=sys.stderr)
        dest.mkdir(parents=True, exist_ok=True)
        _copy_local_gold(sibling, dest)
        return dest

    print(f"fetch gold data from GitHub → {dest}/", file=sys.stderr)
    return download_gold_data(dest)
