"""Upload trained checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _hub_token(explicit: str | None = None) -> str:
    token = explicit or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to upload to Hugging Face Hub")
    return token


def _hub_private() -> bool:
    raw = os.getenv("HF_PRIVATE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def upload_checkpoint(
    output_dir: Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool | None = None,
) -> str:
    """Upload adapter + train_meta.json to a Hugging Face model repo."""
    from huggingface_hub import HfApi

    output_dir = output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    meta_path = output_dir / "train_meta.json"
    if not adapter_dir.is_dir():
        raise SystemExit(f"missing adapter directory: {adapter_dir}")

    resolved_token = _hub_token(token)
    is_private = _hub_private() if private is None else private
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=is_private, token=resolved_token)

    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hf_repo_id"] = repo_id
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    readme_path = output_dir / "README.md"
    if not readme_path.is_file():
        base_model = meta.get("base_model", "unknown")
        readme_path.write_text(
            "\n".join(
                [
                    "---",
                    f"base_model: {base_model}",
                    "library_name: peft",
                    "pipeline_tag: text-generation",
                    "---",
                    "",
                    f"# {repo_id}",
                    "",
                    "LoRA adapter from HuntAI router DPO training (`layer-router-dpo-v1`).",
                    "",
                    f"Base model: `{base_model}`",
                    "",
                    "Load with PEFT / vLLM `--enable-lora`.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        token=resolved_token,
        commit_message=f"Upload router DPO checkpoint ({output_dir.name})",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"uploaded checkpoint -> {url}")
    return url
