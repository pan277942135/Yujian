from app.pipeline_contract import CROP_CLASSIFIER_V1, WHOLE_IMAGE_V1, validate_pipeline_type
from app.training_api import TrainingCreate, _params


def test_training_defaults_to_crop_classifier_without_removing_legacy_mode():
    payload = TrainingCreate(dataset_version="DS_CROP_M1_v0.1")
    assert payload.pipeline_type == CROP_CLASSIFIER_V1
    assert validate_pipeline_type(WHOLE_IMAGE_V1) == WHOLE_IMAGE_V1
    assert _params(payload, "MODEL_CROP_M1_v0.1")["pipeline_type"] == CROP_CLASSIFIER_V1


def test_unknown_pipeline_is_rejected():
    try:
        validate_pipeline_type("ORIGINALS_ONLY")
    except ValueError as exc:
        assert "unsupported pipeline_type" in str(exc)
    else:
        raise AssertionError("unsupported pipeline should be rejected")
