from app.training_api import CROP_CLASSIFIER_V1, WHOLE_IMAGE_V1, release_gate_training_allowed


def test_legacy_release_gate_helper_is_compatibility_only():
    assert release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "PASS"})
    assert release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "PARTIAL_PASS"})
    assert release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "FAIL"})
    assert release_gate_training_allowed(CROP_CLASSIFIER_V1, None)


def test_whole_image_training_keeps_legacy_behavior():
    assert release_gate_training_allowed(WHOLE_IMAGE_V1, None)
    assert release_gate_training_allowed("WHOLE_IMAGE_V1", {"final_release_gate": "PARTIAL_PASS"})
