from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fish_portrait_uses_existing_completion_lab_contract():
    portrait_js = (ROOT / "app/static/js/fish_portrait.js").read_text(encoding="utf-8")
    portrait_html = (ROOT / "app/templates/platform/lab/fish_portrait.html").read_text(encoding="utf-8")
    completion_lab = (ROOT / "app/fish_completion_lab.py").read_text(encoding="utf-8")

    assert "COMPLETION_LAB_PREPARE_ENDPOINT = '/api/debug/fish-completion-lab/prepare'" in portrait_js
    assert "/debug/fish-completion-lab" in portrait_html
    assert '@router.post("/api/debug/fish-completion-lab/prepare")' in completion_lab
    assert "detect(source)" in completion_lab
    assert "generate_fish_cutout(source, primary.box)" in completion_lab

    # The Portrait page is an adapter only; Detector/SAM/mask generation stay
    # in the existing Fish Completion Lab implementation.
    assert "generate_fish_cutout" not in portrait_js
    assert "detect(" not in portrait_js
