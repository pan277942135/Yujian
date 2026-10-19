#!/usr/bin/env python3
"""Verify the exact detector bundle and Case A source before replay.

This read-only helper checks local byte identities only. It never fetches from
GCS and never writes model or dataset artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

CASE_A_SOURCE_SHA256 = "2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe"
CASE_A_SOURCE_SIZE = (1440, 1920)
ANDROID_BUNDLE_SHA256 = "246ddacaf89ca7ecc9c64a47d3e12c5ee9088d92c763a38626778505ae8c15ee"
HISTORICAL_ONNX_SHA256 = "12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f"
REQUIRED_MEMBERS = {
    "fish_detector_yolox_nano_v0_1.onnx",
    "detector_metadata.json",
    "recognition_pipeline_v1.json",
    "golden_cases.json",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_bundle(bundle_path: Path) -> dict:
    raw_bundle = bundle_path.read_bytes()
    bundle_sha = sha256_bytes(raw_bundle)
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_MEMBERS - names)
        if missing:
            raise ValueError(f"detector bundle is missing required files: {missing}")
        onnx = archive.read("fish_detector_yolox_nano_v0_1.onnx")
        metadata_bytes = archive.read("detector_metadata.json")
        contract_bytes = archive.read("recognition_pipeline_v1.json")
        golden_bytes = archive.read("golden_cases.json")
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    contract = json.loads(contract_bytes.decode("utf-8"))
    golden = json.loads(golden_bytes.decode("utf-8"))
    onnx_sha = sha256_bytes(onnx)
    return {
        "bundle": {
            "path_disclosed": False,
            "sha256": bundle_sha,
            "bytes": len(raw_bundle),
            "expected_sha256": ANDROID_BUNDLE_SHA256,
            "matches_expected": bundle_sha == ANDROID_BUNDLE_SHA256,
        },
        "onnx": {
            "sha256": onnx_sha,
            "bytes": len(onnx),
            "metadata_sha256": metadata.get("onnx_sha256"),
            "metadata_bytes": metadata.get("onnx_bytes"),
            "expected_historical_sha256": HISTORICAL_ONNX_SHA256,
            "matches_metadata": onnx_sha == metadata.get("onnx_sha256"),
            "matches_metadata_bytes": len(onnx) == int(metadata.get("onnx_bytes") or -1),
            "matches_historical_case_a_sha": onnx_sha == HISTORICAL_ONNX_SHA256,
        },
        "metadata": {
            "sha256": sha256_bytes(metadata_bytes),
            "model_version": metadata.get("model_version"),
            "model_family": metadata.get("model_family"),
            "dataset_version": metadata.get("dataset_version"),
            "git_commit": metadata.get("git_commit"),
            "training": metadata.get("training"),
            "dataset_report": metadata.get("dataset_report"),
            "onnx_contract": metadata.get("onnx"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        },
        "pipeline_contract": contract,
        "golden_manifest": {
            "sha256": sha256_bytes(golden_bytes),
            "model_version": golden.get("model_version"),
            "dataset_version": golden.get("dataset_version"),
            "onnx_sha256": golden.get("onnx_sha256"),
            "onnx_bytes": golden.get("onnx_bytes"),
            "input_shape": golden.get("input_shape"),
            "case_count": len(golden.get("cases") or []),
            "matches_bundle_onnx": golden.get("onnx_sha256") == onnx_sha,
            "matches_bundle_bytes": golden.get("onnx_bytes") == len(onnx),
        },
    }


def inspect_case_a_source(image_path: Path, expected_sha256: str = CASE_A_SOURCE_SHA256) -> dict:
    raw = image_path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_sha256:
        raise ValueError(f"source SHA mismatch: expected {expected_sha256}, got {actual_sha}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - app runtime installs Pillow
        raise RuntimeError("Pillow is required to verify source image dimensions and EXIF") from exc
    with Image.open(image_path) as image:
        dimensions = image.size
        exif_orientation = int(image.getexif().get(274, 1))
    if dimensions != CASE_A_SOURCE_SIZE:
        raise ValueError(f"source dimensions mismatch: expected {CASE_A_SOURCE_SIZE}, got {dimensions}")
    if exif_orientation != 1:
        raise ValueError(f"source EXIF orientation mismatch: expected 1, got {exif_orientation}")
    return {
        "sha256": actual_sha,
        "bytes": len(raw),
        "dimensions": list(dimensions),
        "exif_orientation": exif_orientation,
        "path_disclosed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--case-a-source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": "CASE_A_ARTIFACT_PREFLIGHT_V1",
        "status": "ARTIFACTS_CHECKED",
        "bundle": inspect_bundle(args.bundle),
        "case_a_source": {
            "status": "BLOCKED_SOURCE_UNAVAILABLE",
            "expected_sha256": CASE_A_SOURCE_SHA256,
            "expected_dimensions": list(CASE_A_SOURCE_SIZE),
        },
    }
    if args.case_a_source:
        result["case_a_source"] = {
            "status": "VERIFIED",
            **inspect_case_a_source(args.case_a_source),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
