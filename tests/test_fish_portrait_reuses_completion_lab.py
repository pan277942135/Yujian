from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fish_portrait_reuses_existing_completion_and_direct_lab_contracts():
    portrait_js = (ROOT / "app/static/js/fish_portrait.js").read_text(encoding="utf-8")
    portrait_html = (ROOT / "app/templates/platform/lab/fish_portrait.html").read_text(encoding="utf-8")
    completion_lab = (ROOT / "app/fish_completion_lab.py").read_text(encoding="utf-8")
    direct_lab = (ROOT / "app/powerpaint_direct_lab.py").read_text(encoding="utf-8")

    assert "COMPLETION_LAB_PREPARE_ENDPOINT = '/api/debug/fish-completion-lab/prepare'" in portrait_js
    assert "POWERPAINT_DIRECT_PREPARE_ENDPOINT = '/api/debug/powerpaint-direct-lab/prepare'" in portrait_js
    assert "/debug/fish-completion-lab" in portrait_html
    assert "/debug/powerpaint-direct-lab" in portrait_html
    assert "fish_preserve_refine_v2" in portrait_js
    assert "fish_preserve_refine_qwen_v1" in portrait_js
    assert "fish_preserve_refine_qwen_v1" in portrait_html
    assert "sam_raw" in completion_lab
    assert "visible_fish_refined" in completion_lab
    assert "Visible Fish Refined" in portrait_html
    assert "visible_add" in portrait_js
    assert "remove" in portrait_js
    assert '@router.post("/api/debug/fish-completion-lab/prepare")' in completion_lab
    assert "detect(source)" in completion_lab
    assert "generate_fish_cutout(source, primary.box)" in completion_lab
    assert '@router.post("/api/debug/powerpaint-direct-lab/prepare")' in direct_lab
    assert "generate_fish_cutout(source, primary.box)" in direct_lab
    assert '"sam_visible": sam_transparent_uri' in direct_lab

    # The Portrait page is an adapter only; Detector/SAM generation stays in
    # the existing lab implementations.
    assert "generate_fish_cutout" not in portrait_js
    assert "detect(" not in portrait_js
