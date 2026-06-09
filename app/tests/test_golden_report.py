"""Unit tests for golden-test report generation (no HTTP)."""

from pathlib import Path

from app.eval.report import generate_report


def test_generate_report_from_fixture(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "router_greeting.csv").write_text(
        "question,expected_route,actual_route,route_match,rewritten_question,actual_answer\n"
        '"Hi","greeting","greeting","true","Hi",""\n'
        '"hack","reject","help","false","hack",""\n',
        encoding="utf-8",
    )
    report_path = result_dir / "router-eval-report-test.md"
    per_file = generate_report(
        report_path=report_path,
        result_dir=result_dir,
        eval_url="http://example/v1/orchestrator/eval/router",
        orchestrator_base="http://example",
        concurrency=4,
        router_prompt_version="router-v2.00",
    )
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "route_match" in text
    assert "router_greeting.csv" in text
    assert len(per_file) == 1
    name, counts = per_file[0]
    assert name == "router_greeting.csv"
    assert counts["true"] == 1
    assert counts["false"] == 1
