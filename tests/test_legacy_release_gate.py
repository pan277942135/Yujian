from app.training_api import CROP_CLASSIFIER_V1, WHOLE_IMAGE_V1, release_gate_training_allowed


def test_crop_training_requires_release_gate_pass():
    assert release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "PASS"})
    assert not release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "PARTIAL_PASS"})
    assert not release_gate_training_allowed(CROP_CLASSIFIER_V1, {"final_release_gate": "FAIL"})
    assert not release_gate_training_allowed(CROP_CLASSIFIER_V1, None)


def test_whole_image_training_keeps_legacy_behavior():
    assert release_gate_training_allowed(WHOLE_IMAGE_V1, None)
    assert release_gate_training_allowed("WHOLE_IMAGE_V1", {"final_release_gate": "PARTIAL_PASS"})
