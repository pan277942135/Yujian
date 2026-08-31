#!/usr/bin/env python3
"""Exercise the deployed-artifact Python detector against audited golden input bytes.

This is intentionally separate from unit tests: ``detect`` loads the ONNX, metadata and
recognition contract from the official GCS prefix through the production runtime path.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

from google.cloud import storage
from PIL import Image, ImageOps

from app.detector_runtime import detect, load_detector, reset_detector_cache
from app.recognition_pipeline import assess_detections, crop_box_pixels, load_contract


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise RuntimeError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def download_bytes(client: storage.Client, uri: str) -> bytes:
    bucket_name, object_name = parse_gs_uri(uri)
    return client.bucket(bucket_name).blob(object_name).download_as_bytes(timeout=180)


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label}: expected={expected:.6f} actual={actual:.6f} tolerance={tolerance:.6f}")


def main() -> None:
    bucket = os.environ.get("GCS_BUCKET", "").strip()
    model_version = os.environ.get("DETECTOR_MODEL_VERSION", "DET_FISH_v0.1").strip()
    if not bucket:
        raise RuntimeError("GCS_BUCKET is required")

    client = storage.Client()
    prefix = f"gs://{bucket}/models/{model_version}"
    document = json.loads(download_bytes(client, f"{prefix}/golden/golden_cases.json").decode("utf-8"))
    if document.get("schema_version") != "DET_FISH_GOLDEN_CASES_v1":
        raise RuntimeError("unexpected golden manifest schema")
    if document.get("model_version") != model_version:
        raise RuntimeError("golden manifest model version mismatch")
    if document.get("pipeline_contract") != load_contract():
        raise RuntimeError("golden manifest recognition contract mismatch")
    tolerance = float(document.get("bbox_tolerance", 0.0))
    crop_tolerance = int(document.get("crop_pixel_tolerance", -1))
    if crop_tolerance != 0:
        raise RuntimeError(f"crop tolerance must be zero, got {crop_tolerance}")

    reset_detector_cache()
    model = load_detector()
    if model.model_version != model_version:
        raise RuntimeError("production detector version mismatch")
    if model.onnx_sha256 != document.get("onnx_sha256"):
        raise RuntimeError("production detector SHA does not match golden manifest")
    if model.onnx_bytes != int(document.get("onnx_bytes") or 0):
        raise RuntimeError("production detector bytes do not match golden manifest")

    expected_statuses = {"READY", "NO_FISH", "INCOMPLETE_FISH", "FISH_TOO_SMALL", "MULTIPLE_FISH"}
    seen: set[str] = set()
    for case in document.get("cases") or []:
        case_id = str(case.get("id") or "")
        raw = download_bytes(client, str(case.get("golden_gcs_uri") or ""))
        if hashlib.sha256(raw).hexdigest() != case.get("source_sha256"):
            raise RuntimeError(f"{case_id}: golden source SHA mismatch")
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        dimensions = case.get("source_dimensions") or {}
        if (image.width, image.height) != (int(dimensions.get("width") or 0), int(dimensions.get("height") or 0)):
            raise RuntimeError(f"{case_id}: golden source dimensions mismatch")

        run = detect(image)
        assessment = assess_detections(run.detections)
        expected_status = str(case.get("expected_status") or "")
        if assessment.status.name != expected_status:
            raise RuntimeError(f"{case_id}: expected {expected_status}, got {assessment.status.name}")
        seen.add(assessment.status.name)

        expected_box = case.get("primary_bbox")
        if expected_box is None:
            if assessment.primary is not None:
                raise RuntimeError(f"{case_id}: unexpected primary bbox")
        else:
            if assessment.primary is None:
                raise RuntimeError(f"{case_id}: missing primary bbox")
            actual_box = assessment.primary.box.normalized()
            for index, value in enumerate((actual_box.x1, actual_box.y1, actual_box.x2, actual_box.y2)):
                assert_close(value, float(expected_box[index]), tolerance, f"{case_id}.bbox[{index}]")

        expected_crop = case.get("crop_pixels")
        if expected_crop is None:
            if assessment.crop_box is not None:
                raise RuntimeError(f"{case_id}: unexpected crop")
        else:
            if assessment.crop_box is None:
                raise RuntimeError(f"{case_id}: missing crop")
            actual_crop = list(crop_box_pixels(assessment.crop_box, image.width, image.height))
            if actual_crop != list(expected_crop):
                raise RuntimeError(f"{case_id}: crop mismatch expected={expected_crop} actual={actual_crop}")
        print(
            f"PRODUCTION_DETECTOR_RUNTIME_CASE_PASS id={case_id} status={assessment.status.name} "
            f"detections={len(run.detections)}",
            flush=True,
        )

    if seen != expected_statuses:
        raise RuntimeError(f"golden status coverage mismatch: expected={sorted(expected_statuses)} actual={sorted(seen)}")
    print(
        "PRODUCTION_DETECTOR_RUNTIME_GATE_PASS "
        f"model={model.model_version} sha256={model.onnx_sha256} bytes={model.onnx_bytes} "
        f"input={model.input_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
