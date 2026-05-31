"""Download router JSONL from layer-orchestrator-v1 GitHub (DPO or SFT)."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.env import REPO_ROOT
from app.method_config import dataset_subdir, normalize_method

DEFAULT_FILES = ("train.jsonl", "val.jsonl", "build-stats.json")
DEFAULT_REPO = "taixingbi/layer-orchestrator-v1"
DEFAULT_REF = "main"
_USER_AGENT = "layer-router-train-v1"


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
    method: str = "dpo",
    ref: str | None = None,
    repo: str | None = None,
    output_subdir: str | None = None,
    files: tuple[str, ...] = DEFAULT_FILES,
) -> Path:
    """Download JSONL files into data_dir; return the directory path."""
    method = normalize_method(method)
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / method)))
    ref = ref or os.environ.get("ORCHESTRATOR_REF", DEFAULT_REF)
    repo = repo or os.environ.get("ORCHESTRATOR_REPO", DEFAULT_REPO)
    output_subdir = output_subdir or dataset_subdir(method)

    base = _base_url(repo, ref, output_subdir)
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{base}/{name}", data_dir / name, name) for name in files]
    for _, _, name in jobs:
        print(f"fetch {name} ({method})", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as pool:
        list(pool.map(lambda job: _download(job[0], job[1]), jobs))
    print(f"dataset -> {data_dir.resolve()}", file=sys.stderr)
    return data_dir


def ensure_dataset(data_dir: Path | None = None, *, method: str = "dpo") -> Path:
    """Download dataset files if train.jsonl is not present."""
    method = normalize_method(method)
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / method)))
    if (data_dir / "train.jsonl").is_file():
        return data_dir
    return download_dataset(data_dir, method=method)
