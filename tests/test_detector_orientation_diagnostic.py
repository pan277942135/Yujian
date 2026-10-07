from app.recognition_pipeline import PipelineStatus
from scripts.diagnose_detector_orientation import _classification


def _matches(original=None, cw=None, ccw=None):
    return {"ORIGINAL": original, "CW90": cw, "CCW90": ccw}


def test_diagnostic_classifies_orientation_recall_gap():
    assert _classification(
        _matches(original=(0.01, 0.8), ccw=(0.35, 0.8)),
        PipelineStatus.NO_FISH,
        0.20,
    ) == "ORIENTATION_RECALL_GAP"


def test_diagnostic_classifies_domain_recall_gap_when_every_view_is_low():
    assert _classification(
        _matches(original=(0.01, 0.8), cw=(0.02, 0.8), ccw=(0.03, 0.8)),
        PipelineStatus.NO_FISH,
        0.20,
    ) == "DOMAIN_RECALL_GAP"


def test_diagnostic_classifies_near_threshold_candidate_separately():
    assert _classification(
        _matches(original=(0.19, 0.8), cw=(0.18, 0.8)),
        PipelineStatus.NO_FISH,
        0.20,
    ) == "THRESHOLD_BORDERLINE"


def test_diagnostic_flags_correct_original_detection_hidden_by_runtime():
    assert _classification(
        _matches(original=(0.25, 0.8), ccw=(0.40, 0.9)),
        PipelineStatus.NO_FISH,
        0.20,
    ) == "DECODE_OR_NMS_DEFECT"
