"""Load router DPO/SFT JSONL from data/output into TRL-ready rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts; raise on invalid lines."""
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _render_prompt(messages: Sequence[Dict[str, str]], *, tokenizer: Any) -> str:
    """Render the shared user prompt once (with generation prompt suffix)."""
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template; use a Qwen instruct checkpoint.")
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def _extract_completion(
    messages: Sequence[Dict[str, str]],
    completion_json: str,
    *,
    tokenizer: Any,
    prompt_text: str,
) -> str:
    """Derive the assistant completion suffix for one chosen/rejected JSON string."""
    content = completion_json.strip()
    assistant_turn = [{"role": "assistant", "content": content}]
    full_text = tokenizer.apply_chat_template(
        list(messages) + assistant_turn,
        tokenize=False,
        add_generation_prompt=False,
    )
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :]
    return content


def _record_to_dpo_row(rec: Dict[str, Any], *, tokenizer: Any) -> Dict[str, str]:
    """Map one DPO JSONL record to {prompt, chosen, rejected}."""
    messages = rec.get("prompt")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("record missing prompt messages")
    chosen_raw = rec.get("chosen")
    rejected_raw = rec.get("rejected")
    if not isinstance(chosen_raw, str) or not isinstance(rejected_raw, str):
        raise ValueError("record missing chosen/rejected strings")
    json.loads(chosen_raw)
    json.loads(rejected_raw)

    prompt_text = _render_prompt(messages, tokenizer=tokenizer)
    return {
        "prompt": prompt_text,
        "chosen": _extract_completion(
            messages, chosen_raw, tokenizer=tokenizer, prompt_text=prompt_text
        ),
        "rejected": _extract_completion(
            messages, rejected_raw, tokenizer=tokenizer, prompt_text=prompt_text
        ),
    }


def records_to_dpo_rows(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    """Map router DPO JSONL records to {prompt, chosen, rejected}."""
    return [_record_to_dpo_row(rec, tokenizer=tokenizer) for rec in records]


def load_dpo_dataset(
    train_path: Path,
    val_path: Optional[Path],
    *,
    tokenizer: Any,
) -> tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """Load train (and optional val) JSONL paths into TRL DPO rows."""
    train_rows = records_to_dpo_rows(load_jsonl(train_path), tokenizer=tokenizer)
    val_rows: Optional[List[Dict[str, str]]] = None
    if val_path and val_path.is_file():
        val_records = load_jsonl(val_path)
        if val_records:
            val_rows = records_to_dpo_rows(val_records, tokenizer=tokenizer)
    return train_rows, val_rows


def _record_to_sft_row(rec: Dict[str, Any], *, tokenizer: Any) -> Dict[str, str]:
    """Map one SFT record to {text} via chat template."""
    messages = rec.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("record missing messages")
    if not any(m.get("role") == "assistant" for m in messages if isinstance(m, dict)):
        raise ValueError("record missing assistant turn")
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template; use a Qwen instruct checkpoint.")
    for turn in messages:
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            json.loads(turn.get("content", ""))
    text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def records_to_sft_rows(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    """Map SFT JSONL records to {text} rows for SFTTrainer."""
    return [_record_to_sft_row(rec, tokenizer=tokenizer) for rec in records]


def load_sft_dataset(
    train_path: Path,
    val_path: Optional[Path],
    *,
    tokenizer: Any,
) -> tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """Load train (and optional val) JSONL paths into TRL SFT rows."""
    train_rows = records_to_sft_rows(load_jsonl(train_path), tokenizer=tokenizer)
    val_rows: Optional[List[Dict[str, str]]] = None
    if val_path and val_path.is_file():
        val_records = load_jsonl(val_path)
        if val_records:
            val_rows = records_to_sft_rows(val_records, tokenizer=tokenizer)
    return train_rows, val_rows
