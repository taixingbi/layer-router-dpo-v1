"""Load router DPO JSONL produced by build_from_gold.py into TRL-ready rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _messages_to_prompt_and_completion(
    messages: Sequence[Dict[str, str]],
    completion_json: str,
    *,
    tokenizer: Any,
) -> Dict[str, str]:
    """Split chat messages + router JSON completion for TRL DPO (prompt / chosen|rejected)."""
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat_template; use a Qwen instruct checkpoint.")

    prompt_text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    assistant_turn = [{"role": "assistant", "content": completion_json.strip()}]
    full_chosen = tokenizer.apply_chat_template(
        list(messages) + assistant_turn,
        tokenize=False,
        add_generation_prompt=False,
    )
    if full_chosen.startswith(prompt_text):
        completion_text = full_chosen[len(prompt_text) :]
    else:
        completion_text = completion_json.strip()

    return {"prompt": prompt_text, "completion": completion_text}


def records_to_dpo_rows(
    records: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    """Map build_from_gold JSONL records to {prompt, chosen, rejected}."""
    out: List[Dict[str, str]] = []
    for rec in records:
        messages = rec.get("prompt")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("record missing prompt messages")
        chosen_raw = rec.get("chosen")
        rejected_raw = rec.get("rejected")
        if not isinstance(chosen_raw, str) or not isinstance(rejected_raw, str):
            raise ValueError("record missing chosen/rejected strings")
        json.loads(chosen_raw)
        json.loads(rejected_raw)

        chosen_row = _messages_to_prompt_and_completion(
            messages, chosen_raw, tokenizer=tokenizer
        )
        rejected_row = _messages_to_prompt_and_completion(
            messages, rejected_raw, tokenizer=tokenizer
        )
        if chosen_row["prompt"] != rejected_row["prompt"]:
            raise ValueError("chosen/rejected prompt mismatch for same record")
        out.append(
            {
                "prompt": chosen_row["prompt"],
                "chosen": chosen_row["completion"],
                "rejected": rejected_row["completion"],
            }
        )
    return out


def load_dpo_dataset(
    train_path: Path,
    val_path: Optional[Path],
    *,
    tokenizer: Any,
) -> tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    train_records = load_jsonl(train_path)
    train_rows = records_to_dpo_rows(train_records, tokenizer=tokenizer)
    val_rows: Optional[List[Dict[str, str]]] = None
    if val_path and val_path.is_file():
        val_records = load_jsonl(val_path)
        if val_records:
            val_rows = records_to_dpo_rows(val_records, tokenizer=tokenizer)
    return train_rows, val_rows
