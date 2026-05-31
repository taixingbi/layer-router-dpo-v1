"""Tests for app/train_metrics.py."""

from __future__ import annotations

from types import SimpleNamespace

from app.train_metrics import collect_train_timings, log_timings_banner, readme_timing_section


def test_collect_train_timings():
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=42))
    result = SimpleNamespace(
        metrics={
            "train_samples_per_second": 0.154321,
            "train_steps_per_second": 0.019876,
        }
    )
    timings = collect_train_timings(trainer, result, wall_sec=312.789)
    assert timings == {
        "train_sec": 312.79,
        "train_samples_per_second": 0.1543,
        "train_steps_per_second": 0.0199,
        "global_step": 42,
    }


def test_collect_train_timings_missing_metrics():
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=1))
    result = SimpleNamespace(metrics={})
    timings = collect_train_timings(trainer, result, wall_sec=10.0)
    assert timings["train_sec"] == 10.0
    assert timings["train_samples_per_second"] is None
    assert timings["global_step"] == 1


def test_log_timings_banner(capsys):
    log_timings_banner(
        "DPO",
        {
            "train_sec": 100.0,
            "train_samples_per_second": 0.5,
            "train_steps_per_second": 0.1,
            "global_step": 10,
        },
    )
    out = capsys.readouterr().out
    assert "TRAINING TIMING — DPO" in out
    assert "wall clock:     100.0s" in out
    assert "samples/sec:    0.5" in out


def test_readme_timing_section():
    lines = readme_timing_section({"train_sec": 312.8, "global_step": 48})
    text = "\n".join(lines)
    assert "## Training timing" in text
    assert "**312.8s**" in text
    assert "Global step" in text


def test_readme_timing_section_empty():
    assert readme_timing_section(None) == []
    assert readme_timing_section({}) == []
