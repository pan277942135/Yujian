from __future__ import annotations

from evaluation.artifact_builder import build_evaluation_artifacts
from app.intelligence_api import build_intelligence_payload, load_evaluation_document


class EmptyDB:
    def scalars(self, _statement):
        return _Rows([])

    def scalar(self, _statement):
        return None

    def get(self, _model, _key):
        return None


class _Rows:
    def __init__(self, _rows):
        self.rows = _rows

    def all(self):
        return list(self.rows)


def test_standard_artifact_bundle_is_joined_for_model_intelligence(tmp_path, monkeypatch):
    artifact_root = tmp_path / "evaluation_artifacts"
    build_evaluation_artifacts(
        {
            "model_version": "MODEL_M1_v0.3",
            "dataset_version": "DS_M1_v0.3",
            "test": {
                "count": 107,
                "accuracy": 0.645,
                "top3_accuracy": 0.879,
                "macro_precision": 0.662,
                "macro_recall": 0.599,
                "macro_f1": 0.601,
                "confusion_matrix": [[6, 3, 2], [3, 7, 0], [1, 0, 6]],
            },
            "classes": ["grass_carp", "common_carp", "black_carp"],
        },
        [
            {"image_id": "P500001", "true_species": "grass_carp", "pred_species": "common_carp", "confidence": 0.62},
            {"image_id": "P500002", "true_species": "common_carp", "pred_species": "grass_carp", "confidence": 0.73},
            {"image_id": "P500003", "true_species": "grass_carp", "pred_species": "black_carp", "confidence": 0.54},
        ],
        artifact_root,
    )
    monkeypatch.setenv("EVALUATION_ARTIFACT_ROOT", str(artifact_root))

    document, source, warning = load_evaluation_document(EmptyDB(), "MODEL_M1_v0.3")
    assert warning is None
    assert source and source.endswith("MODEL_M1_v0.3/metrics.json")
    assert document["labels"] == ["grass_carp", "common_carp", "black_carp"]
    assert document["confusion_matrix"][0] == [6, 3, 2]
    assert len(document["samples"]) == 3
    assert len(document["errors"]) == 3

    payload = build_intelligence_payload(EmptyDB(), evaluation_document=document)
    assert payload["model"]["evaluation_status"] == "READY"
    pair = next(
        row
        for row in payload["confusion_report"]["top_confusions"]
        if row["true_species"] == "grass_carp" and row["pred_species"] == "common_carp"
    )
    assert (pair["true_species"], pair["pred_species"], pair["priority"]) == ("grass_carp", "common_carp", "P0")
    assert payload["evaluation_artifacts"]["contract_version"] == "evaluation-artifact-v1"
