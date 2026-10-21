import pytest

from trainer.detector_training_contract import validate_training_target_versions


def test_v03_training_requires_exact_dataset_model_pair_and_commit():
    validate_training_target_versions(
        "DET_DS_v0.3",
        "DET_FISH_v0.3",
        "DET_DS_v0.3",
        "5ab51176d4640047b7df8c55f79b517b52ef3bfb",
    )


@pytest.mark.parametrize(
    "dataset,model,gate,commit",
    [
        ("", "DET_FISH_v0.3", "DET_DS_v0.3", "a" * 40),
        ("DET_DS_v0.1", "DET_FISH_v0.1", "DET_DS_v0.1", "a" * 40),
        ("DET_DS_v0.3", "DET_FISH_v0.2", "DET_DS_v0.3", "a" * 40),
        ("DET_DS_v0.3", "DET_FISH_v0.3", "DET_DS_v0.2", "a" * 40),
        ("DET_DS_v0.3", "DET_FISH_v0.3", "DET_DS_v0.3", "unknown"),
    ],
)
def test_training_contract_rejects_implicit_or_mismatched_versions(dataset, model, gate, commit):
    with pytest.raises(ValueError):
        validate_training_target_versions(dataset, model, gate, commit)
