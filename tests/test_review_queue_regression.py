from types import SimpleNamespace
from io import BytesIO

from PIL import Image

from app import main
from app.main import FeedbackEvent, review_queue
from app.recognition_pipeline import BBox, Detection


def test_review_queue_has_feedback_event_dependency_imported():
    """The review queue enriches rows from the latest feedback event."""
    assert review_queue.__globals__["FeedbackEvent"] is FeedbackEvent


def test_review_bbox_reidentification_uses_parity_normalized_xywh(monkeypatch):
    image_bytes = BytesIO()
    Image.new("RGB", (100, 100), "white").save(image_bytes, format="PNG")
    monkeypatch.setattr(main, "_read_uri", lambda _uri: (image_bytes.getvalue(), None))
    monkeypatch.setattr(main, "normalize_android_source", lambda image: image.copy())
    detection = Detection(confidence=0.9, box=BBox(0.1, 0.2, 0.8, 0.9))
    monkeypatch.setattr(
        main,
        "detect",
        lambda _image: SimpleNamespace(
            detections=(detection,),
            model_version="DET_FISH_v0.1",
            latency_ms=12.3,
        ),
    )

    candidate, model_version, assessment, latency_ms = main._detect_candidate_bbox(
        SimpleNamespace(gcs_uri="gs://bucket/image.png")
    )

    assert candidate == [0.1, 0.2, 0.7, 0.7]
    assert model_version == "DET_FISH_v0.1"
    assert assessment == "ready"
    assert latency_ms == 12.3
