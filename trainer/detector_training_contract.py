"""Fail-closed version contract for DET_FISH training jobs."""

from __future__ import annotations

import re


TRAINING_VERSION_PAIRS = {
    "DET_DS_v0.2": "DET_FISH_v0.2",
    "DET_DS_v0.3": "DET_FISH_v0.3",
}


def validate_training_target_versions(
    dataset_version: str,
    model_version: str,
    reviewed_dataset_version: str,
    app_git_commit: str,
) -> None:
    """Require an explicit, supported dataset/model pair and exact source SHA."""
    if not dataset_version:
        raise ValueError("DETECTOR_DATASET_VERSION is required; refusing implicit dataset defaults")
    if not model_version:
        raise ValueError("DETECTOR_MODEL_VERSION is required; refusing implicit model defaults")
    expected_model = TRAINING_VERSION_PAIRS.get(dataset_version)
    if expected_model is None:
        raise ValueError(f"unsupported reviewed detector dataset version: {dataset_version}")
    if dataset_version != reviewed_dataset_version:
        raise ValueError(
            "runtime dataset version does not match the dataset gate version: "
            f"job={dataset_version} gate={reviewed_dataset_version or '<missing>'}"
        )
    if model_version != expected_model:
        raise ValueError(
            f"{dataset_version} must publish {expected_model}, not {model_version}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", app_git_commit):
        raise ValueError("APP_GIT_COMMIT must be the exact 40-character lowercase Git SHA")
