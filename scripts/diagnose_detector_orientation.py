#!/usr/bin/env python3
"""Inspect raw DET_FISH ONNX candidates for original/CW90/CCW90 image views.

This diagnostic never changes production thresholds or inference behavior. Raw rows
are decoded with min_confidence=0 and without NMS; production-pass status is computed
separately with the frozen weak threshold and NMS contract.

Ground-truth JSON is optional and maps each image basename to a normalized xyxy box:
{"case.jpg": [0.2, 0.1, 0.7, 0.9]}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnxruntime as ort
from PIL import Image, ImageOps

from app.detector_runtime import (
    DETECTOR_ORIENTATIONS,
    _prepare_yolox_input,
    decode_yolox_candidates,
    decode_yolox_output,
    map_box_to_original,
    normalize_android_source,
)
from app.recognition_pipeline import BBox, PipelineStatus, assess_detections, load_contract

THRESHOLD_BORDERLINE_MARGIN = 0.02


def _iou(left: BBox, right: BBox) -> float:
    a, b = left.normalized(), right.normalized()
    x = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    y = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    overlap = x * y
    union = a.area_ratio + b.area_ratio - overlap
    return overlap / union if union > 0.0 else 0.0


def _ground_truth(document: dict, image_path: Path) -> BBox | None:
    value = document.get(image_path.name)
    if isinstance(value, dict):
        value = value.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    return BBox(*(float(item) for item in value)).normalized()


def _classification(
    matches: dict[str, tuple[float, float] | None],
    original_status: PipelineStatus,
    weak_confidence: float,
) -> str:
    original = matches["ORIGINAL"]
    rotated = [matches[key] for key in ("CW90", "CCW90") if matches[key] is not None]
    if original is not None and original[0] >= weak_confidence and original_status is PipelineStatus.NO_FISH:
        return "DECODE_OR_NMS_DEFECT"
    if original is not None and original[0] < weak_confidence and any(
        item[0] >= weak_confidence for item in rotated
    ):
        return "ORIENTATION_RECALL_GAP"
    all_correct = [item[0] for item in matches.values() if item is not None]
    best_correct_confidence = max(all_correct, default=0.0)
    if weak_confidence - THRESHOLD_BORDERLINE_MARGIN <= best_correct_confidence < weak_confidence:
        return "THRESHOLD_BORDERLINE"
    return "DOMAIN_RECALL_GAP"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--ground-truth-json", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    model_bytes = args.onnx.read_bytes()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(model_bytes).hexdigest()
    if metadata.get("onnx_sha256") != actual_sha:
        raise SystemExit(f"ONNX SHA mismatch: metadata={metadata.get('onnx_sha256')} actual={actual_sha}")
    if int(metadata.get("onnx_bytes") or 0) != len(model_bytes):
        raise SystemExit("ONNX byte-size mismatch with detector metadata")
    if metadata.get("model_family") != "YOLOX_NANO":
        raise SystemExit("diagnostic requires the production YOLOX_NANO artifact")

    contract = load_contract()
    input_size = int(contract["detector"]["input_size"])
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    if list(session.get_inputs()[0].shape) != [1, 3, input_size, input_size]:
        raise SystemExit(f"unexpected ONNX input shape: {session.get_inputs()[0].shape}")
    gt_document = (
        json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
        if args.ground_truth_json
        else {}
    )

    print(f"model_version={metadata.get('model_version')}")
    print(f"onnx_sha256={actual_sha}")
    print(f"weak_confidence={contract['detector']['weak_confidence']} nms_iou={contract['detector']['nms_iou']}")

    for image_path in args.images:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        normalized = normalize_android_source(image)
        image.close()
        gt = _ground_truth(gt_document, image_path)
        weak_confidence = float(contract["detector"]["weak_confidence"])
        matches: dict[str, tuple[float, float] | None] = {}
        original_status = PipelineStatus.NO_FISH
        print(f"\n## {image_path.name} ({normalized.width}x{normalized.height})")
        print("| orientation | rank | objectness | fish_probability | confidence | bbox_original_xyxy | area_ratio | IoU_GT |")
        print("|---|---:|---:|---:|---:|---|---:|---:|")

        for orientation in DETECTOR_ORIENTATIONS:
            if orientation == "ORIGINAL":
                view = normalized
            elif orientation == "CW90":
                view = normalized.transpose(Image.Transpose.ROTATE_270)
            else:
                view = normalized.transpose(Image.Transpose.ROTATE_90)
            try:
                tensor, scale, _, _ = _prepare_yolox_input(view, input_size)
                output = session.run(None, {input_name: tensor})[0]
                all_candidates = decode_yolox_candidates(
                    output,
                    scale=scale,
                    source_width=view.width,
                    source_height=view.height,
                    min_confidence=0.0,
                    top_k=output.shape[-2],
                )
                production_detections = decode_yolox_output(
                    output,
                    scale=scale,
                    source_width=view.width,
                    source_height=view.height,
                    nms_iou=float(contract["detector"]["nms_iou"]),
                    min_confidence=float(contract["detector"]["weak_confidence"]),
                )
                mapped_production = tuple(
                    type(detection)(
                        confidence=detection.confidence,
                        box=map_box_to_original(detection.box, orientation),
                        class_name=detection.class_name,
                    )
                    for detection in production_detections
                )
                if orientation == "ORIGINAL":
                    original_status = assess_detections(mapped_production, contract).status

                scored: list[tuple[float, object]] = []
                for candidate in all_candidates:
                    mapped_box = map_box_to_original(candidate.box, orientation)
                    overlap = _iou(mapped_box, gt) if gt else 0.0
                    if gt:
                        scored.append((overlap, candidate))
                for candidate in all_candidates[: args.top_k]:
                    mapped_box = map_box_to_original(candidate.box, orientation)
                    overlap = _iou(mapped_box, gt) if gt else 0.0
                    print(
                        f"| {orientation} | {candidate.rank} | {candidate.objectness:.6f} | "
                        f"{candidate.fish_probability:.6f} | {candidate.confidence:.6f} | "
                        f"[{mapped_box.x1:.6f},{mapped_box.y1:.6f},{mapped_box.x2:.6f},{mapped_box.y2:.6f}] | "
                        f"{mapped_box.area_ratio:.6f} | {overlap:.3f} |"
                    )
                matched = [item for item in scored if item[0] >= 0.50]
                if matched:
                    best_correct = max(matched, key=lambda item: item[1].confidence)
                    matches[orientation] = (best_correct[1].confidence, best_correct[0])
                else:
                    matches[orientation] = None
            finally:
                if view is not normalized:
                    view.close()

        if gt:
            classification = _classification(matches, original_status, weak_confidence)
            print(f"classification={classification} original_gate={original_status.value}")
            print(f"threshold_borderline_band=[{weak_confidence - THRESHOLD_BORDERLINE_MARGIN:.2f},{weak_confidence})")
            print(f"best_correct_confidence_by_orientation={json.dumps(matches, sort_keys=True)}")
        else:
            print("classification=UNCLASSIFIED_NO_GROUND_TRUTH; provide --ground-truth-json")
        normalized.close()


if __name__ == "__main__":
    main()
