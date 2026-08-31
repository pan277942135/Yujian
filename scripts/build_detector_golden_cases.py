#!/usr/bin/env python3
"""Publish real, audited DET_DS_v0.1 images as detector parity golden cases.

The script never creates images or detector outputs. Each fixture is copied from the
reviewed detector bootstrap manifest only after the official DET_FISH_v0.1 ONNX returns
the recorded quality-gate status on that exact source image.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

# Running `python scripts/<name>.py` places scripts/ rather than the repository root
# on sys.path. Resolve backend modules deterministically before importing app.*.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnxruntime as ort
from google.cloud import storage
from PIL import Image, ImageOps

from app.detector_runtime import _prepare_yolox_input, decode_yolox_output
from app.recognition_pipeline import Detection, assess_detections, crop_box_pixels, load_contract


REQUIRED_STATUSES = ("READY", "NO_FISH", "INCOMPLETE_FISH", "FISH_TOO_SMALL", "MULTIPLE_FISH")


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def download_bytes(client: storage.Client, uri: str) -> bytes:
    bucket_name, object_name = parse_gs_uri(uri)
    return client.bucket(bucket_name).blob(object_name).download_as_bytes(timeout=180)


def detections_document(detections: tuple[Detection, ...]) -> list[dict[str, Any]]:
    return [
        {
            "confidence": round(float(item.confidence), 6),
            "class_name": item.class_name,
            "bbox": [
                round(item.box.normalized().x1, 6),
                round(item.box.normalized().y1, 6),
                round(item.box.normalized().x2, 6),
                round(item.box.normalized().y2, 6),
            ],
        }
        for item in detections
    ]


def manifest_priority(entry: dict[str, Any]) -> tuple[int, str]:
    """Prefer audited candidates likely to cover every production gate before fallbacks."""
    status = str(entry.get("presence_status") or "")
    boxes = list(entry.get("boxes") or [])
    touches_edge = any(bool(box.get("touches_edge")) for box in boxes)
    has_small_box = any(
        len(box.get("bbox_normalized") or []) == 4
        and max(0.0, float(box["bbox_normalized"][2]) - float(box["bbox_normalized"][0]))
        * max(0.0, float(box["bbox_normalized"][3]) - float(box["bbox_normalized"][1]))
        < 0.08
        for box in boxes
    )
    priority = 5
    if status == "no_fish":
        priority = 0
    elif status == "multi_fish":
        priority = 1
    elif touches_edge:
        priority = 2
    elif has_small_box:
        priority = 3
    elif status == "single_fish":
        priority = 4
    return priority, str(entry.get("image_asset_id") or "")


def run_detector(
    session: ort.InferenceSession,
    image: Image.Image,
    contract: dict,
) -> tuple[tuple[Detection, ...], float, int, int]:
    input_size = int(contract["detector"]["input_size"])
    tensor, scale, draw_width, draw_height = _prepare_yolox_input(image, input_size)
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    detections = decode_yolox_output(
        output,
        scale=scale,
        source_width=image.width,
        source_height=image.height,
        nms_iou=float(contract["detector"]["nms_iou"]),
        min_confidence=float(contract["detector"]["weak_confidence"]),
    )
    return detections, scale, draw_width, draw_height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dataset-version", default="DET_DS_v0.1")
    parser.add_argument("--model-version", default="DET_FISH_v0.1")
    parser.add_argument("--max-images", type=int, default=600)
    args = parser.parse_args()

    client = storage.Client()
    model_prefix = f"gs://{args.bucket}/models/{args.model_version}"
    metadata = json.loads(download_bytes(client, f"{model_prefix}/detector_metadata.json").decode("utf-8"))
    artifact_contract = json.loads(download_bytes(client, f"{model_prefix}/recognition_pipeline_v1.json").decode("utf-8"))
    contract = load_contract()
    if artifact_contract != contract:
        raise RuntimeError("golden generation blocked: artifact recognition contract differs from source contract")
    if metadata.get("model_version") != args.model_version or metadata.get("dataset_version") != args.dataset_version:
        raise RuntimeError("golden generation blocked: detector metadata version mismatch")

    onnx_bytes = download_bytes(client, f"{model_prefix}/fish_detector_yolox_nano_v0_1.onnx")
    onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    if onnx_sha256 != metadata.get("onnx_sha256"):
        raise RuntimeError("golden generation blocked: detector ONNX SHA256 mismatch")
    if len(onnx_bytes) != int(metadata.get("onnx_bytes") or 0):
        raise RuntimeError("golden generation blocked: detector ONNX byte-size mismatch")

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_shape = list(session.get_inputs()[0].shape)
    expected_input = list((metadata.get("onnx") or {}).get("input_shape") or [])
    if input_shape != expected_input:
        raise RuntimeError(f"golden generation blocked: ONNX input shape {input_shape} != {expected_input}")

    manifest_uri = f"gs://{args.bucket}/detector-datasets/{args.dataset_version}/bootstrap_manifest.json"
    manifest = json.loads(download_bytes(client, manifest_uri).decode("utf-8"))
    candidates = [entry for entry in manifest if not entry.get("skipped") and entry.get("gcs_uri")]
    candidates.sort(key=manifest_priority)

    selected: dict[str, dict[str, Any]] = {}
    attempted = 0
    for entry in candidates:
        if len(selected) == len(REQUIRED_STATUSES) or attempted >= args.max_images:
            break
        attempted += 1
        source_uri = str(entry["gcs_uri"])
        try:
            image_bytes = download_bytes(client, source_uri)
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            detections, scale, draw_width, draw_height = run_detector(session, image, contract)
            assessment = assess_detections(detections, contract)
        except Exception as exc:
            print(f"GOLDEN_CASE_SKIP image_asset_id={entry.get('image_asset_id')} error={exc}", flush=True)
            continue
        status = assessment.status.name
        if status not in REQUIRED_STATUSES or status in selected:
            continue

        suffix = Path(parse_gs_uri(source_uri)[1]).suffix.lower() or ".jpg"
        golden_name = f"{assessment.status.value}{suffix}"
        golden_object = f"models/{args.model_version}/golden/{golden_name}"
        client.bucket(args.bucket).blob(golden_object).upload_from_string(image_bytes)
        crop_pixels = (
            crop_box_pixels(assessment.crop_box, image.width, image.height) if assessment.crop_box is not None else None
        )
        selected[status] = {
            "id": assessment.status.value,
            "expected_status": status,
            "source_dataset_version": args.dataset_version,
            "source_image_asset_id": entry.get("image_asset_id"),
            "source_presence_status": entry.get("presence_status"),
            "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "source_dimensions": {"width": image.width, "height": image.height},
            "golden_gcs_uri": f"gs://{args.bucket}/{golden_object}",
            "detector_input": {
                "scale": scale,
                "draw_width": draw_width,
                "draw_height": draw_height,
            },
            "detections": detections_document(detections),
            "primary_bbox": (
                detections_document((assessment.primary,))[0]["bbox"] if assessment.primary is not None else None
            ),
            "crop_pixels": list(crop_pixels) if crop_pixels is not None else None,
        }
        print(
            f"GOLDEN_CASE_SELECTED status={status} image_asset_id={entry.get('image_asset_id')} source={source_uri}",
            flush=True,
        )

    missing = [status for status in REQUIRED_STATUSES if status not in selected]
    if missing:
        raise RuntimeError(
            "GOLDEN_CASE_GATE_FAIL: audited DET_DS images did not yield required actual detector statuses "
            f"{missing}; attempted={attempted}"
        )

    document = {
        "schema_version": "DET_FISH_GOLDEN_CASES_v1",
        "pipeline_contract": contract,
        "model_version": args.model_version,
        "dataset_version": args.dataset_version,
        "onnx_sha256": onnx_sha256,
        "onnx_bytes": len(onnx_bytes),
        "input_shape": input_shape,
        "bbox_tolerance": 0.004,
        "crop_pixel_tolerance": 0,
        "cases": [selected[status] for status in REQUIRED_STATUSES],
    }
    output_object = f"models/{args.model_version}/golden/golden_cases.json"
    client.bucket(args.bucket).blob(output_object).upload_from_string(
        json.dumps(document, ensure_ascii=False, indent=2), content_type="application/json"
    )
    print(
        json.dumps(
            {
                "status": "DET_FISH_GOLDEN_CASE_GATE_PASS",
                "golden_cases": f"gs://{args.bucket}/{output_object}",
                "statuses": list(selected),
                "attempted": attempted,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
