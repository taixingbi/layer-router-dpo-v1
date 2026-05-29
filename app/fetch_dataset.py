"""Download default DPO JSONL from layer-orchestrator-v1 GitHub (used by train)."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.env import REPO_ROOT

DEFAULT_FILES = ("train.jsonl", "val.jsonl", "build-stats.json")
DEFAULT_REPO = "taixingbi/layer-orchestrator-v1"
DEFAULT_REF = "main"
DEFAULT_OUTPUT_SUBDIR = "dpo-router/output"
_USER_AGENT = "layer-router-dpo-v1"


def _base_url(repo: str, ref: str, output_subdir: str) -> str:
    """Build the raw GitHub base URL for dataset files."""
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{output_subdir.strip('/')}"


def _download(url: str, dest: Path) -> None:
    """Fetch a single file from url and write it to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            dest.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"fetch failed {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"fetch failed {url}: {exc.reason}") from exc


def download_dataset(
    data_dir: Path | None = None,
    *,
    ref: str | None = None,
    repo: str | None = None,
    output_subdir: str | None = None,
    files: tuple[str, ...] = DEFAULT_FILES,
) -> Path:
    """Download JSONL files into data_dir; return the directory path."""
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
    ref = ref or os.environ.get("ORCHESTRATOR_DPO_REF", DEFAULT_REF)
    repo = repo or os.environ.get("ORCHESTRATOR_DPO_REPO", DEFAULT_REPO)
    output_subdir = output_subdir or os.environ.get("ORCHESTRATOR_DPO_SUBDIR", DEFAULT_OUTPUT_SUBDIR)

    base = _base_url(repo, ref, output_subdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{base}/{name}", data_dir / name, name) for name in files]
    for _, _, name in jobs:
        print(f"fetch {name}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as pool:
        list(pool.map(lambda job: _download(job[0], job[1]), jobs))
    print(f"dataset -> {data_dir.resolve()}", file=sys.stderr)
    return data_dir


def ensure_dataset(data_dir: Path | None = None) -> Path:
    """Download dataset files if train.jsonl is not present."""
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
    if (data_dir / "train.jsonl").is_file():
        return data_dir
    return download_dataset(data_dir)
