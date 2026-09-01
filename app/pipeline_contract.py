"""Names and safety checks for classifier input pipelines."""

WHOLE_IMAGE_V1 = "WHOLE_IMAGE_V1"
CROP_CLASSIFIER_V1 = "CROP_CLASSIFIER_V1"
PIPELINE_TYPES = {WHOLE_IMAGE_V1, CROP_CLASSIFIER_V1}


def validate_pipeline_type(value: str | None) -> str:
    pipeline = str(value or WHOLE_IMAGE_V1).strip().upper()
    if pipeline not in PIPELINE_TYPES:
        raise ValueError(f"unsupported pipeline_type: {pipeline}")
    return pipeline
