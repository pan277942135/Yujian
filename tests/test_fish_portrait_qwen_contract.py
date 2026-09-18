from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_page_mode_and_worker_contract_are_present():
    route = (ROOT / "app/platform/routes/portrait.py").read_text(encoding="utf-8")
    client = (ROOT / "app/qwen_refine_worker_client.py").read_text(encoding="utf-8")
    javascript = (ROOT / "app/static/js/fish_portrait.js").read_text(encoding="utf-8")
    html = (ROOT / "app/templates/platform/lab/fish_portrait.html").read_text(encoding="utf-8")
    completion = (ROOT / "app/fish_completion_lab.py").read_text(encoding="utf-8")
    worker = (ROOT / "workers/fish-qwen-refine-worker/worker.py").read_text(encoding="utf-8")
    service = (ROOT / "workers/fish-qwen-refine-worker/fish-qwen-refine-worker.service").read_text(encoding="utf-8")

    assert "fish_preserve_refine_qwen_v1" in route
    assert "fish_preserve_refine_qwen_v1" in client
    assert "QWEN_MODE" in javascript
    assert "鱼体保真补全（Qwen V1）" in html
    assert "COMPLETION_LAB_PREPARE_ENDPOINT = '/api/debug/fish-completion-lab/prepare'" in javascript
    assert '"sam_visible": f"/api/debug/fish-completion-lab/media/{test_id}/sam-visible"' in completion
    assert "POST /refine" in worker or '@app.post("/refine"' in worker
    assert 'port=8002' in worker
    assert '--port 8002' in service
    assert "/portrait/generate" not in client
