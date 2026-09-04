#!/usr/bin/env python3
"""Build a byte-verified Android distribution bundle from official DET_FISH GCS objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from google.cloud import storage


REQUIRED_CASE_IDS = {"ready", "no_fish", "incomplete_fish", "fish_too_small", "multiple_fish"}
MODEL_FILES = (
    "fish_detector_yolox_nano_v0_1.onnx",
    "detector_metadata.json",
    "recognition_pipeline_v1.json",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def suffix(uri: str) -> str:
    extension = Path(uri.rsplit("/", 1)[-1]).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise RuntimeError(f"unsupported golden image extension: {uri}")
    return extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-version", default="DET_FISH_v0.1")
    parser.add_argument("--output", default="det_fish_v0_1_android_bundle.zip")
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    model_prefix = f"models/{args.model_version}"

    objects: dict[str, bytes] = {}
    for filename in MODEL_FILES:
        objects[filename] = bucket.blob(f"{model_prefix}/{filename}").download_as_bytes(timeout=180)
    metadata = json.loads(objects["detector_metadata.json"].decode("utf-8"))
    contract = json.loads(objects["recognition_pipeline_v1.json"].decode("utf-8"))
    onnx = objects["fish_detector_yolox_nano_v0_1.onnx"]
    if metadata.get("model_version") != args.model_version or metadata.get("dataset_version") != "DET_DS_v0.1":
        raise RuntimeError("detector metadata version mismatch")
    if metadata.get("model_family") != "YOLOX_NANO":
        raise RuntimeError("detector model family mismatch")
    if metadata.get("onnx_sha256") != sha256(onnx) or int(metadata.get("onnx_bytes") or 0) != len(onnx):
        raise RuntimeError("detector ONNX metadata integrity mismatch")
    if contract.get("contract_version") != "RECOGNITION_PIPELINE_v1":
        raise RuntimeError("recognition contract mismatch")

    checkpoint = bucket.blob(f"{model_prefix}/best_ckpt.pth")
    if not checkpoint.exists(client):
        raise RuntimeError("official DET_FISH best_ckpt.pth is missing")
    checkpoint.reload(client)
    if int(checkpoint.size or 0) <= 0:
        raise RuntimeError("official DET_FISH best_ckpt.pth is empty")

    manifest_bytes = bucket.blob(f"{model_prefix}/golden/golden_cases.json").download_as_bytes(timeout=180)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema_version") != "DET_FISH_GOLDEN_CASES_v1":
        raise RuntimeError("detector golden manifest schema mismatch")
    if manifest.get("model_version") != args.model_version or manifest.get("onnx_sha256") != sha256(onnx):
        raise RuntimeError("detector golden manifest artifact mismatch")

    golden_prefix = f"gs://{args.bucket}/{model_prefix}/golden/"
    golden: dict[str, bytes] = {}
    seen: set[str] = set()
    for case in manifest.get("cases") or []:
        case_id = str(case.get("id") or "")
        uri = str(case.get("golden_gcs_uri") or "")
        if not case_id or case_id in seen or not uri.startswith(golden_prefix):
            raise RuntimeError(f"invalid golden case: {case!r}")
        seen.add(case_id)
        image = bucket.blob(uri[5:].split("/", 1)[1]).download_as_bytes(timeout=180)
        if sha256(image) != case.get("source_sha256"):
            raise RuntimeError(f"golden source SHA mismatch for {case_id}")
        golden[f"golden/{case_id}{suffix(uri)}"] = image
    if seen != REQUIRED_CASE_IDS:
        raise RuntimeError(f"golden case coverage mismatch: {sorted(seen)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for filename, data in objects.items():
            archive.writestr(filename, data)
        archive.writestr("golden_cases.json", manifest_bytes)
        for filename, data in sorted(golden.items()):
            archive.writestr(filename, data)
    print(
        "DET_FISH_ANDROID_BUNDLE_PASS "
        f"bundle={output} bundle_sha256={sha256(output.read_bytes())} "
        f"onnx_sha256={sha256(onnx)} onnx_bytes={len(onnx)} best_ckpt_bytes={checkpoint.size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
