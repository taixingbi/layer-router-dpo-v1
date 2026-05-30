"""Upload trained checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class HubUploadError(RuntimeError):
    """Raised when checkpoint upload to Hugging Face Hub fails."""


def _hub_token(explicit: str | None = None) -> str:
    token = explicit or os.getenv("HF_TOKEN")
    if not token:
        raise HubUploadError("HF_TOKEN is required to upload to Hugging Face Hub")
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
        raise HubUploadError("could not resolve Hugging Face username from HF_TOKEN")
    resolved = f"{owner}/layer-router-dpo-v1"
    print(f"HF_REPO_ID not set, using token owner -> {resolved}", file=sys.stderr)
    return resolved


def _ensure_repo(api, target_repo: str, token: str, *, private: bool) -> None:
    from huggingface_hub.errors import HfHubHTTPError

    if api.repo_exists(repo_id=target_repo, repo_type="model", token=token):
        return
    try:
        api.create_repo(
            target_repo,
            exist_ok=True,
            repo_type="model",
            private=private,
            token=token,
        )
    except HfHubHTTPError as exc:
        owner = target_repo.split("/", 1)[0]
        me = api.whoami().get("name", "?")
        if me == owner:
            raise HubUploadError(
                f"cannot create Hugging Face repo {target_repo}: {exc}\n"
                f"HF_TOKEN for '{me}' lacks Write permission. "
                f"Create a new token at https://huggingface.co/settings/tokens with Write access."
            ) from exc
        raise HubUploadError(
            f"cannot create Hugging Face repo {target_repo}: {exc}\n"
            f"HF_TOKEN user is '{me}' but repo namespace is '{owner}'.\n"
            f"Use a Write token for '{owner}', or set HF_REPO_ID={me}/layer-router-dpo-v1"
        ) from exc


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
        raise HubUploadError(f"missing adapter directory: {adapter_dir}")

    resolved_token = _hub_token(token)
    is_private = _hub_private() if private is None else private
    api = HfApi(token=resolved_token)
    target_repo = resolve_repo_id(api, repo_id or os.getenv("HF_REPO_ID"))

    _ensure_repo(api, target_repo, resolved_token, private=is_private)

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

    try:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=target_repo,
            repo_type="model",
            token=resolved_token,
            commit_message=f"Upload router DPO checkpoint ({output_dir.name})",
        )
    except HfHubHTTPError as exc:
        raise HubUploadError(
            f"upload to {target_repo} failed: {exc}\n"
            "Ensure HF_TOKEN has Write permission for this repo."
        ) from exc

    url = f"https://huggingface.co/{target_repo}"
    print(f"uploaded checkpoint -> {url}")
    return url
