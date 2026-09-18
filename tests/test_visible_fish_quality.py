import numpy as np

from app.visible_fish_quality import analyze_visible_fish_quality, quality_status


def test_visible_fish_quality_requires_a_refinement_step():
    raw = np.zeros((32, 48), dtype=bool)
    raw[10:16, 8:40] = True
    report = analyze_visible_fish_quality(raw, raw, [4, 4, 44, 24], raw.shape)

    assert report["visible_fish_quality"] == "WARNING"
    assert report["quality_gate_passed"] is False
    assert report["raw_sam_area"] == int(raw.sum())
    assert report["refined_visible_area"] == int(raw.sum())
    assert "VISIBLE_REFINEMENT_REQUIRED" in report["quality_reasons"]


def test_visible_fish_quality_accepts_connected_refined_fish():
    raw = np.zeros((32, 48), dtype=bool)
    raw[10:14, 8:40] = True
    refined = raw.copy()
    refined[14:19, 8:40] = True
    report = analyze_visible_fish_quality(refined & raw, refined, [4, 4, 44, 24], raw.shape)

    assert report["visible_fish_quality"] == "GOOD"
    assert report["quality_gate_passed"] is True
    assert report["connected_components"] == 1
    assert report["largest_component_ratio"] == 1.0
    assert report["bbox_coverage_ratio"] > 0.2


def test_visible_fish_quality_rejects_disconnected_or_empty_input():
    raw = np.zeros((32, 48), dtype=bool)
    refined = np.zeros_like(raw)
    refined[4:8, 4:8] = True
    refined[14:18, 18:22] = True
    refined[24:28, 32:36] = True
    report = analyze_visible_fish_quality(raw, refined, [0, 0, 48, 32], raw.shape)

    assert report["visible_fish_quality"] == "INVALID"
    assert report["connected_components"] == 3
    assert report["largest_component_ratio"] < 0.55
    assert quality_status(report) == "INVALID"
