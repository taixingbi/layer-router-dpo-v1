from app.build.paths import golden_result_dir, router_model_slug


def test_router_model_slug():
    assert router_model_slug("router-qwen2.5-7b-sft-v1.00") == "router-qwen2.5-7b-sft-v1.00"
    assert router_model_slug("org/model:v1") == "org__model__v1"


def test_golden_result_dir_with_model(monkeypatch):
    monkeypatch.delenv("RESULT_DIR", raising=False)
    p = golden_result_dir("router-qwen2.5-7b-sft-v1.00")
    assert p.name == "router-qwen2.5-7b-sft-v1.00"
    assert p.parent.name == "result"
    assert p.parent.parent.name == "data"


def test_golden_result_dir_flat(monkeypatch):
    monkeypatch.delenv("ROUTER_MODEL", raising=False)
    monkeypatch.delenv("RESULT_DIR", raising=False)
    p = golden_result_dir("")
    assert p.name == "result"
    assert p.parent.name == "data"
