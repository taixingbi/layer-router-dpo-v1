"""Collect and print training timing metrics."""

from __future__ import annotations

from typing import Any


def collect_train_timings(trainer: Any, train_result: Any, *, wall_sec: float) -> dict[str, float | int | None]:
    """Build a timings dict from Trainer output and wall-clock duration."""
    metrics = getattr(train_result, "metrics", None) or {}
    return {
        "train_sec": round(wall_sec, 2),
        "train_samples_per_second": _round_metric(metrics.get("train_samples_per_second")),
        "train_steps_per_second": _round_metric(metrics.get("train_steps_per_second")),
        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
    }


def _round_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def log_timings_banner(label: str, timings: dict[str, float | int | None]) -> None:
    """Print a highlighted timing summary to stdout (visible in deploy logs)."""
    width = 60
    bar = "=" * width
    lines = [
        "",
        bar,
        f"  TRAINING TIMING — {label}",
        bar,
        f"  wall clock:     {timings.get('train_sec')}s",
    ]
    samples = timings.get("train_samples_per_second")
    if samples is not None:
        lines.append(f"  samples/sec:    {samples}")
    steps = timings.get("train_steps_per_second")
    if steps is not None:
        lines.append(f"  steps/sec:      {steps}")
    lines.append(f"  global_step:    {timings.get('global_step')}")
    lines.extend([bar, ""])
    print("\n".join(lines), flush=True)


def readme_timing_section(timings: dict[str, float | int | None] | None) -> list[str]:
    """Markdown lines for Hub README training timing table."""
    if not timings:
        return []
    rows = [
        "",
        "## Training timing",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Train time | **{timings.get('train_sec')}s** |",
    ]
    samples = timings.get("train_samples_per_second")
    if samples is not None:
        rows.append(f"| Samples/sec | {samples} |")
    steps = timings.get("train_steps_per_second")
    if steps is not None:
        rows.append(f"| Steps/sec | {steps} |")
    rows.append(f"| Global step | {timings.get('global_step')} |")
    rows.append("")
    return rows
