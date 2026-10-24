#!/usr/bin/env python3
"""Replay DET_FISH ONNX candidates for ORIGINAL/CW90/CCW90 views.

The raw probe uses confidence threshold 0 and no NMS. Production gating is
computed separately with the frozen contract. An optional source SHA guard
prevents accidentally replaying a resized, re-encoded, or unrelated fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image

from app.detector_runtime import (
    DETECTOR_ORIENTATIONS,
    _assessment_priority,
    _assessment_rank_score,
    _prepare_yolox_input,
    _selection_reason,
    decode_yolox_candidates,
    decode_yolox_output,
    map_box_to_original,
)
from app.recognition_pipeline import BBox, PipelineStatus, assess_detections, load_contract

THRESHOLD_BORDERLINE_MARGIN = 0.02
CORRECT_CANDIDATE_IOU = 0.50
CASE_A_SOURCE_SHA256 = "2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe"
CASE_A_SOURCE_SIZE = (1440, 1920)


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
    best = max((item[0] for item in matches.values() if item is not None), default=0.0)
    if weak_confidence - THRESHOLD_BORDERLINE_MARGIN <= best < weak_confidence:
        return "THRESHOLD_BORDERLINE"
    return "DOMAIN_RECALL_GAP"


def _source_identity(path: Path, expected_sha256: str | None) -> dict:
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
        exif_orientation = int(image.getexif().get(274, 1))
    if expected_sha256 and actual_sha != expected_sha256:
        raise SystemExit(f"{path.name}: source SHA mismatch: expected={expected_sha256} actual={actual_sha}")
    if expected_sha256 == CASE_A_SOURCE_SHA256 and (width, height) != CASE_A_SOURCE_SIZE:
        raise SystemExit(
            f"{path.name}: Case A dimensions mismatch: expected={CASE_A_SOURCE_SIZE} actual={(width, height)}"
        )
    if exif_orientation != 1:
        raise SystemExit(f"{path.name}: EXIF orientation {exif_orientation}; expected 1 for this raw replay")
    return {
        "sha256": actual_sha,
        "bytes": len(raw),
        "format": image_format,
        "width": width,
        "height": height,
        "exif_orientation": exif_orientation,
        "orientation_normalization": {
            "applied": False,
            "status": "identity_exif_orientation_1",
        },
        "local_path_disclosed": False,
    }


def _array_summary(value: np.ndarray) -> dict:
    array = np.asarray(value)
    rows = array[0] if array.ndim == 3 else array
    finite = rows[np.isfinite(rows)]
    column_ranges = []
    for column in range(rows.shape[-1]):
        cells = rows[:, column]
        cells = cells[np.isfinite(cells)]
        column_ranges.append({
            "column": column,
            "min": float(cells.min()) if cells.size else None,
            "max": float(cells.max()) if cells.size else None,
        })
    probabilities = rows[:, 4:6]
    probabilities = probabilities[np.isfinite(probabilities)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_output_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "raw_output_bytes": int(array.nbytes),
        "finite_value_count": int(finite.size),
        "non_finite_value_count": int(rows.size - finite.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "column_ranges": column_ranges,
        "objectness_and_class_probability_within_0_1": bool(
            probabilities.size
            and np.all(probabilities >= 0.0)
            and np.all(probabilities <= 1.0)
        ),
    }


def _tensor_summary(tensor: np.ndarray) -> dict:
    value = np.asarray(tensor)
    channel_statistics = []
    for channel in range(value.shape[1]):
        plane = value[0, channel]
        channel_statistics.append({
            "channel": channel,
            "mean": float(plane.mean()),
            "std": float(plane.std()),
            "min": float(plane.min()),
            "max": float(plane.max()),
        })
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "channel_means": [float(value[0, channel].mean()) for channel in range(value.shape[1])],
        "channel_statistics": channel_statistics,
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _candidate_record(candidate, box: BBox, overlap: float | None) -> dict:
    return {
        "rank": candidate.rank,
        "objectness": candidate.objectness,
        "fish_probability": candidate.fish_probability,
        "confidence": candidate.confidence,
        "bbox_original_xyxy": [box.x1, box.y1, box.x2, box.y2],
        "area_ratio": box.area_ratio,
        "gt_iou": overlap,
    }


def _write_raw_output_artifact(output: np.ndarray, path: Path) -> dict:
    raw_output = np.ascontiguousarray(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, raw_output, allow_pickle=False)
    artifact_bytes = path.read_bytes()
    return {
        "filename": path.name,
        "file_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "file_bytes": len(artifact_bytes),
        "raw_output_sha256": hashlib.sha256(raw_output.tobytes(order="C")).hexdigest(),
        "shape": list(raw_output.shape),
        "dtype": str(raw_output.dtype),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--ground-truth-json", type=Path)
    parser.add_argument("--expected-image-sha256")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        help="optionally save exact per-orientation ONNX outputs as lossless .npy files",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    model_bytes = args.onnx.read_bytes()
    metadata_bytes = args.metadata.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    actual_sha = hashlib.sha256(model_bytes).hexdigest()
    if metadata.get("onnx_sha256") != actual_sha:
        raise SystemExit(f"ONNX SHA mismatch: metadata={metadata.get('onnx_sha256')} actual={actual_sha}")
    if int(metadata.get("onnx_bytes") or 0) != len(model_bytes):
        raise SystemExit("ONNX byte-size mismatch with detector metadata")
    if metadata.get("model_family") != "YOLOX_NANO":
        raise SystemExit("diagnostic requires the production YOLOX_NANO artifact")

    source_records = {
        path: _source_identity(path, args.expected_image_sha256) for path in args.images
    }
    contract = load_contract()
    input_size = int(contract["detector"]["input_size"])
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()
    if len(session_inputs) != 1 or len(session_outputs) != 1:
        raise SystemExit(
            f"unexpected ONNX input/output count: inputs={len(session_inputs)} outputs={len(session_outputs)}"
        )
    input_name = session_inputs[0].name
    output_name = session_outputs[0].name
    input_shape = list(session_inputs[0].shape)
    output_shape = list(session_outputs[0].shape)
    if input_shape != [1, 3, input_size, input_size]:
        raise SystemExit(f"unexpected ONNX input shape: {input_shape}")
    expected_output_shape = (metadata.get("onnx") or {}).get("output_shape")
    if expected_output_shape and output_shape != expected_output_shape:
        raise SystemExit(
            f"ONNX output shape differs from metadata: runtime={output_shape} metadata={expected_output_shape}"
        )
    gt_document = (
        json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
        if args.ground_truth_json
        else {}
    )

    print(f"model_version={metadata.get('model_version')}")
    print(f"onnx_sha256={actual_sha}")
    print(f"weak_confidence={contract['detector']['weak_confidence']} nms_iou={contract['detector']['nms_iou']}")
    replay = {
        "schema_version": "CASE_A_ORIENTATION_REPLAY_V1",
        "status": "REPLAYED",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_git_sha": _git_sha(),
        "command_template": "python scripts/diagnose_detector_orientation.py --onnx <onnx> --metadata <metadata> --ground-truth-json <gt> --expected-image-sha256 <expected> --json-out <output> <image>",
        "runtime": {
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "model": {
            "model_version": metadata.get("model_version"),
            "model_family": metadata.get("model_family"),
            "dataset_version": metadata.get("dataset_version"),
            "onnx_sha256": actual_sha,
            "onnx_bytes": len(model_bytes),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "input_name": input_name,
            "input_shape": input_shape,
            "output_name": output_name,
            "output_shape": output_shape,
            "output_contract": (metadata.get("onnx") or {}).get("output_contract"),
        },
        "production_gate": {
            "weak_confidence": float(contract["detector"]["weak_confidence"]),
            "strong_confidence": float(contract["detector"]["strong_confidence"]),
            "nms_iou": float(contract["detector"]["nms_iou"]),
            "min_primary_area_ratio": float(contract["quality_gate"]["min_primary_area_ratio"]),
            "candidate_probe_confidence": 0.0,
            "candidate_probe_nms": False,
            "candidate_probe_top_k": args.top_k,
            "correct_candidate_iou_threshold": CORRECT_CANDIDATE_IOU,
        },
        "images": [],
    }

    for image_path in args.images:
        with Image.open(image_path) as source:
            normalized = source.convert("RGB")
        if normalized.size != (source_records[image_path]["width"], source_records[image_path]["height"]):
            raise SystemExit(f"{image_path.name}: decoder changed source dimensions")
        gt = _ground_truth(gt_document, image_path)
        image_record = {"source": source_records[image_path], "ground_truth_xyxy": None, "orientations": []}
        if gt:
            image_record["ground_truth_xyxy"] = [gt.x1, gt.y1, gt.x2, gt.y2]
        matches: dict[str, tuple[float, float] | None] = {}
        assessments = {}
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
                tensor, scale, draw_width, draw_height = _prepare_yolox_input(view, input_size)
                output = session.run([output_name], {input_name: tensor})[0]
                raw_output_artifact = None
                if args.raw_output_dir:
                    source_sha_prefix = source_records[image_path]["sha256"][:12]
                    raw_output_path = args.raw_output_dir / (
                        f"{source_sha_prefix}.{orientation.lower()}.npy"
                    )
                    raw_output_artifact = _write_raw_output_artifact(output, raw_output_path)
                all_candidates = decode_yolox_candidates(
                    output,
                    scale=scale,
                    source_width=view.width,
                    source_height=view.height,
                    min_confidence=0.0,
                    top_k=int(output.shape[-2]),
                )
                production = decode_yolox_output(
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
                    for detection in production
                )
                assessment = assess_detections(mapped_production, contract)
                assessments[orientation] = assessment
                all_scored = []
                for candidate in all_candidates:
                    box = map_box_to_original(candidate.box, orientation)
                    overlap = _iou(box, gt) if gt else None
                    all_scored.append((overlap, candidate, box))
                for overlap, candidate, box in all_scored[:args.top_k]:
                    overlap_display = f"{overlap:.3f}" if overlap is not None else "n/a"
                    print(
                        f"| {orientation} | {candidate.rank} | {candidate.objectness:.6f} | "
                        f"{candidate.fish_probability:.6f} | {candidate.confidence:.6f} | "
                        f"[{box.x1:.6f},{box.y1:.6f},{box.x2:.6f},{box.y2:.6f}] | "
                        f"{box.area_ratio:.6f} | {overlap_display} |"
                    )
                matched = [
                    item for item in all_scored
                    if gt and item[0] is not None and item[0] >= CORRECT_CANDIDATE_IOU
                ]
                best_correct = max(matched, key=lambda item: item[1].confidence) if matched else None
                matches[orientation] = (
                    (best_correct[1].confidence, best_correct[0]) if best_correct else None
                )
                production_boxes = [
                    {
                        "confidence": item.confidence,
                        "bbox_original_xyxy": [item.box.x1, item.box.y1, item.box.x2, item.box.y2],
                        "area_ratio": item.box.area_ratio,
                    }
                    for item in mapped_production
                ]
                image_record["orientations"].append({
                    "orientation": orientation,
                    "view_dimensions": [view.width, view.height],
                    "input_scale": scale,
                    "input_draw_dimensions": [draw_width, draw_height],
                    "input_tensor": _tensor_summary(tensor),
                    "raw_output_summary": _array_summary(output),
                    "raw_output_artifact": raw_output_artifact,
                    "production_gate": {
                        "status": assessment.status.value,
                        "detection_count_after_threshold_and_nms": len(mapped_production),
                        "primary_confidence": assessment.primary.confidence if assessment.primary else None,
                        "primary_bbox_original_xyxy": (
                            [assessment.primary.box.x1, assessment.primary.box.y1,
                             assessment.primary.box.x2, assessment.primary.box.y2]
                            if assessment.primary else None
                        ),
                        "primary_area_ratio": assessment.primary.box.area_ratio if assessment.primary else None,
                        "detections": production_boxes,
                    },
                    "best_correct_candidate": (
                        _candidate_record(best_correct[1], best_correct[2], best_correct[0])
                        if best_correct else None
                    ),
                    "top_candidates": [
                        _candidate_record(candidate, box, overlap)
                        for overlap, candidate, box in all_scored[:args.top_k]
                    ],
                })
            finally:
                if view is not normalized:
                    view.close()

        if gt:
            original_status = assessments["ORIGINAL"].status
            weak = float(contract["detector"]["weak_confidence"])
            classification = _classification(matches, original_status, weak)
            print(f"classification={classification} original_gate={original_status.value}")
            print(f"threshold_borderline_band=[{weak - THRESHOLD_BORDERLINE_MARGIN:.2f},{weak})")
            print(f"best_correct_confidence_by_orientation={json.dumps(matches, sort_keys=True)}")
            image_record["classification"] = classification
        original_status = assessments["ORIGINAL"].status
        if original_status is PipelineStatus.NO_FISH:
            attempts = list(DETECTOR_ORIENTATIONS)
            eligible = [
                (orientation, assessment)
                for orientation, assessment in assessments.items()
                if assessment.status is not PipelineStatus.NO_FISH
            ]
            selected = max(
                eligible,
                key=lambda item: (
                    _assessment_priority(item[1]),
                    _assessment_rank_score(item[1]),
                    -DETECTOR_ORIENTATIONS.index(item[0]),
                ),
                default=None,
            )
            selected_orientation = selected[0] if selected else "NONE"
            selected_assessment = selected[1] if selected else None
            reason = _selection_reason(selected_assessment) if selected_assessment else "NO_FISH_AFTER_BOUNDED_RETRIES"
        else:
            attempts = ["ORIGINAL"]
            selected_orientation = "ORIGINAL"
            selected_assessment = assessments["ORIGINAL"]
            reason = "ORIGINAL_ASSESSMENT_PRESERVED"
        image_record["bounded_retry_replay"] = {
            "policy_version": "DETECTOR_ORIENTATION_RETRY_v2",
            "attempts": attempts,
            "selected_attempt": selected_orientation,
            "selection_reason": reason,
            "selected_gate_status": selected_assessment.status.value if selected_assessment else "no_fish",
        }
        replay["images"].append(image_record)
        normalized.close()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
