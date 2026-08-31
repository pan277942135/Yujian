from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from trainer import build_detector_dataset as dataset


def test_vertices_convert_to_normalized_detector_box():
    box = dataset._vertices_to_box([
        {"x": 0.8, "y": 0.7},
        {"x": 0.2, "y": 0.3},
    ])
    assert box is not None
    assert (box.x1, box.y1, box.x2, box.y2) == (0.2, 0.3, 0.8, 0.7)


def test_deterministic_split_uses_batch_and_image_identity():
    image = SimpleNamespace(batch_id="BATCH_001", image_id="IMG_001")
    assert dataset._split_key(image) == dataset._split_key(image)
    assert dataset._split_key(image) in {"train", "val", "test"}


def test_failure_diagnostic_is_published_without_replacing_root_exception(monkeypatch):
    uploads: list[tuple[str, str]] = []

    class Blob:
        def __init__(self, name: str):
            self.name = name

        def upload_from_string(self, body: str, content_type: str):
            uploads.append((self.name, body))
            assert content_type == "application/json"

    class Bucket:
        def blob(self, name: str):
            return Blob(name)

    class Client:
        def bucket(self, name: str):
            assert name == "test-bucket"
            return Bucket()

    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("DETECTOR_DATASET_VERSION", "DET_DS_v0.1")
    try:
        raise RuntimeError("database not reachable")
    except RuntimeError as exc:
        with patch.object(dataset.storage, "Client", return_value=Client()):
            dataset._publish_failure(exc)

    assert len(uploads) == 1
    object_name, body = uploads[0]
    assert object_name == "detector-datasets/DET_DS_v0.1/bootstrap_failure.json"
    doc = json.loads(body)
    assert doc["error_type"] == "RuntimeError"
    assert doc["error"] == "database not reachable"
    assert "RuntimeError: database not reachable" in doc["traceback"]
