from app.det_fish_console import _dataset_gate, _execution_status, ALLOWED_MODEL_VERSIONS


def _ready_report():
    return {
        "dataset_version": "DET_DS_v0.2",
        "status": "FROZEN_READY_FOR_TRAINING",
        "source": "accepted_bbox_human_review",
        "human_review_verified": True,
        "held_out_exclusions_verified": True,
        "split_integrity": "PASS",
        "negative_coverage_verified": True,
        "manifest_sha256": "a" * 64,
        "negatives": 12,
        "splits": {
            "train": {"images": 100, "annotations": 100},
            "val": {"images": 10, "annotations": 10},
            "test": {"images": 10, "annotations": 10},
        },
    }


def test_dataset_gate_accepts_only_frozen_human_reviewed_dataset():
    result = _dataset_gate(_ready_report(), archive_exists=True, manifest_verified=True)
    assert result["ready"] is True
    assert result["blockers"] == []
    assert result["counts"]["train_images"] == 100


def test_dataset_gate_fails_closed_for_missing_archive_and_provenance():
    report = _ready_report()
    report["source"] = "FishPresenceResult/google-vision-presence-v0.4"
    report["held_out_exclusions_verified"] = False
    result = _dataset_gate(report, archive_exists=False, manifest_verified=False)
    assert result["ready"] is False
    assert any("人工 accepted_bbox" in item for item in result["blockers"])
    assert any("留出集排除" in item for item in result["blockers"])
    assert any("冻结数据包" in item for item in result["blockers"])


def test_dataset_gate_requires_independent_test_and_negative_coverage():
    report = _ready_report()
    report["negative_coverage_verified"] = False
    report["splits"]["test"]["images"] = 0
    result = _dataset_gate(report, archive_exists=True, manifest_verified=True)
    assert result["ready"] is False
    assert any("无鱼负样本" in item for item in result["blockers"])
    assert any("独立测试集" in item for item in result["blockers"])


def test_execution_status_distinguishes_running_and_failed():
    assert _execution_status({"runningCount": 1}) == "RUNNING"
    assert _execution_status({"completionTime": "2026-10-20T00:00:00Z", "failedCount": 1}) == "FAILED"
    assert _execution_status({"completionTime": "2026-10-20T00:00:00Z", "succeededCount": 1}) == "SUCCEEDED"


def test_only_supported_detector_versions_can_be_selected():
    assert ALLOWED_MODEL_VERSIONS == {"DET_FISH_v0.1", "DET_FISH_v0.2"}

def test_dataset_gate_requires_manifest_bytes_to_match_report_sha():
    result = _dataset_gate(_ready_report(), archive_exists=True, manifest_verified=False)
    assert result["ready"] is False
    assert any("manifest.json" in item for item in result["blockers"])
