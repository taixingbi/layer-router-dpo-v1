"""Upload trained checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _hub_token(explicit: str | None = None) -> str:
    token = explicit or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to upload to Hugging Face Hub")
    return token


def _hub_private() -> bool:
    raw = os.getenv("HF_PRIVATE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_repo_id(api, repo_id: str | None) -> str:
    """Use HF_REPO_ID when set, otherwise {token_owner}/layer-router-dpo-v1."""
    if repo_id and repo_id.strip():
        return repo_id.strip()
    who = api.whoami()
    owner = who.get("name") or who.get("fullname")
    if not owner:
        raise SystemExit("could not resolve Hugging Face username from HF_TOKEN")
    resolved = f"{owner}/layer-router-dpo-v1"
    print(f"HF_REPO_ID not set, using token owner -> {resolved}", file=sys.stderr)
    return resolved


def upload_checkpoint(
    output_dir: Path,
    repo_id: str | None = None,
    *,
    token: str | None = None,
    private: bool | None = None,
) -> str:
    """Upload adapter + train_meta.json to a Hugging Face model repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    output_dir = output_dir.resolve()
    adapter_dir = output_dir / "adapter"
    meta_path = output_dir / "train_meta.json"
    if not adapter_dir.is_dir():
        raise SystemExit(f"missing adapter directory: {adapter_dir}")

    resolved_token = _hub_token(token)
    is_private = _hub_private() if private is None else private
    api = HfApi(token=resolved_token)
    target_repo = resolve_repo_id(api, repo_id or os.getenv("HF_REPO_ID"))

    if not api.repo_exists(repo_id=target_repo, repo_type="model", token=resolved_token):
        try:
            api.create_repo(
                target_repo,
                exist_ok=True,
                repo_type="model",
                private=is_private,
                token=resolved_token,
            )
        except HfHubHTTPError as exc:
            owner = target_repo.split("/", 1)[0]
            who = api.whoami()
            me = who.get("name", "?")
            raise SystemExit(
                f"cannot create Hugging Face repo {target_repo}: {exc}\n"
                f"HF_TOKEN user is '{me}' but repo namespace is '{owner}'.\n"
                f"Use a Write token for '{owner}', or set HF_REPO_ID={me}/layer-router-dpo-v1"
            ) from exc

    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hf_repo_id"] = target_repo
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
                    f"# {target_repo}",
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
        repo_id=target_repo,
        repo_type="model",
        token=resolved_token,
        commit_message=f"Upload router DPO checkpoint ({output_dir.name})",
    )
    url = f"https://huggingface.co/{target_repo}"
    print(f"uploaded checkpoint -> {url}")
    return url
