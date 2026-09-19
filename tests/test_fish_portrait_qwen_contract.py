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
    assert '"sam_raw": f"/api/debug/fish-completion-lab/media/{test_id}/sam-raw"' in completion
    assert '"visible_fish_refined": f"/api/debug/fish-completion-lab/media/{test_id}/visible-fish-refined"' in completion
    assert "VISIBLE_FISH_QUALITY_GATE_BLOCKED" in route
    assert '"quality_gate_passed": quality == "GOOD"' in (ROOT / "app/visible_fish_quality.py").read_text(encoding="utf-8")
    assert "VISIBLE_REFINEMENT_REQUIRED" not in (ROOT / "app/visible_fish_quality.py").read_text(encoding="utf-8")
    assert "elapsed_ms" in route
    assert "visible_fish_refined_uri" in route
    assert '"input_source": "visible_fish_refined"' in client
    assert "POST /refine" in worker or '@app.post("/refine"' in worker
    assert 'port=int(os.getenv("PORT", "8002"))' in worker
    assert '"input_source": "visible_fish_refined"' in worker
    assert '"elapsed_ms"' in worker
    assert '--port 8002' in service
    assert "/portrait/generate" not in client
    assert "SAM Raw" in html
    assert "Visible Fish Refined" in html
    assert "portraitVisibleCorrection" in javascript
