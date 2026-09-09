from app import powerpaint_direct_lab as direct


def test_direct_lab_prompt_and_experiment_are_fixed():
    assert direct.DIRECT_VERSION == "POWERPAINT_DIRECT_V0.1"
    assert "Extract the fish from the image." in direct.DIRECT_PROMPT
    assert "Do not change the fish species." in direct.DIRECT_PROMPT
    assert "case_label" not in direct.DIRECT_PROMPT


def test_direct_lab_progress_has_only_direct_stages():
    report = {
        "input": {"filename": "IMG00103"},
        "detector": {"model": "DET_FISH_v0.1"},
        "sam": {"quality": "GOOD"},
        "worker": {"status": "WORKER_EXECUTED", "result_uri": "gs://bucket/result.png"},
        "timings": {"input_decode_ms": 1, "detector_ms": 2, "sam_ms": 3, "worker_ms": 4},
    }
    stages = direct._progress(report)
    assert [stage["stage"] for stage in stages] == ["input", "detector", "sam", "powerpaint"]
    assert stages[-1]["status"] == "WORKER_EXECUTED"
