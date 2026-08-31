from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from google.cloud import storage

from trainer.yolox_fish_exp import Exp


PRETRAIN_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            destination = (target / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        tf.extractall(target)


def download_gcs(bucket_name: str, object_name: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(str(target))


def upload_gcs(bucket_name: str, object_name: str, source: Path, content_type: str | None = None) -> str:
    blob = storage.Client().bucket(bucket_name).blob(object_name)
    blob.upload_from_filename(str(source), content_type=content_type)
    return f"gs://{bucket_name}/{object_name}"


def require_dataset(report: dict) -> None:
    splits = report.get("splits") or {}
    train = splits.get("train") or {}
    val = splits.get("val") or {}
    test = splits.get("test") or {}
    min_train_boxes = int(os.environ.get("DETECTOR_MIN_TRAIN_BOXES", "30"))
    min_val_boxes = int(os.environ.get("DETECTOR_MIN_VAL_BOXES", "3"))
    min_test_images = int(os.environ.get("DETECTOR_MIN_TEST_IMAGES", "1"))
    min_negatives = int(os.environ.get("DETECTOR_MIN_NEGATIVES", "1"))
    train_boxes = int(train.get("annotations") or 0)
    val_boxes = int(val.get("annotations") or 0)
    train_images = int(train.get("images") or 0)
    if train_boxes < min_train_boxes or val_boxes < min_val_boxes:
        raise RuntimeError(
            "DETECTOR_DATASET_GATE_FAIL: "
            f"train_boxes={train_boxes} (need {min_train_boxes}), "
            f"val_boxes={val_boxes} (need {min_val_boxes})"
        )
    if train_images < 40:
        raise RuntimeError(f"DETECTOR_DATASET_GATE_FAIL: train_images={train_images} < 40")
    test_images = int(test.get("images") or 0)
    negatives = int(report.get("negatives") or 0)
    if test_images < min_test_images:
        raise RuntimeError(
            "DETECTOR_DATASET_GATE_FAIL: "
            f"test_images={test_images} (need {min_test_images})"
        )
    if negatives < min_negatives:
        raise RuntimeError(
            "DETECTOR_DATASET_GATE_FAIL: "
            f"no_fish_negatives={negatives} (need {min_negatives})"
        )


def run_training(dataset_root: Path, pretrain: Path, output_root: Path) -> Path:
    env = dict(os.environ)
    env["DETECTOR_DATASET_ROOT"] = str(dataset_root)
    env["DETECTOR_OUTPUT_DIR"] = str(output_root)
    batch = os.environ.get("DETECTOR_BATCH_SIZE", "16")
    command = [
        sys.executable,
        "/opt/YOLOX/tools/train.py",
        "-f",
        "/app/trainer/yolox_fish_exp.py",
        "-d",
        "1",
        "-b",
        batch,
        "--fp16",
        "-c",
        str(pretrain),
        "-expn",
        "fish_detector_yolox_nano",
    ]
    subprocess.run(command, check=True, env=env)
    experiment_dir = output_root / "fish_detector_yolox_nano"
    checkpoint = experiment_dir / "best_ckpt.pth"
    if checkpoint.exists():
        return checkpoint
    raise RuntimeError(f"YOLOX training completed but no best checkpoint found in {experiment_dir}")


def evaluate_checkpoint(checkpoint: Path) -> dict:
    """Measure the actual exported checkpoint with COCO AP50 and AP50:95 metrics."""
    if not torch.cuda.is_available():
        raise RuntimeError("DET_FISH evaluation requires CUDA; Cloud Run GPU was not attached")

    exp = Exp()
    model = exp.get_model()
    state = torch.load(checkpoint, map_location="cpu")
    model_state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)
    model.cuda().eval()

    batch_size = int(os.environ.get("DETECTOR_BATCH_SIZE", "16"))
    evaluator = exp.get_evaluator(batch_size=batch_size, is_distributed=False)
    ap50_95, ap50, summary = exp.eval(model, evaluator, is_distributed=False, half=True)
    return {
        "ap50": float(ap50),
        "ap50_95": float(ap50_95),
        "summary": str(summary or ""),
    }


def export_onnx(checkpoint: Path, output: Path) -> dict:
    from torch import nn
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    exp = Exp()
    model = exp.get_model()
    state = torch.load(checkpoint, map_location="cpu")
    model_state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True

    dummy = torch.zeros(1, 3, exp.test_size[0], exp.test_size[1], dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    model_doc = onnx.load(str(output))
    onnx.checker.check_model(model_doc)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    result = session.run(None, {input_info.name: np.zeros((1, 3, 416, 416), dtype=np.float32)})[0]
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[-1] != 6:
        raise RuntimeError(f"unexpected YOLOX ONNX output shape: {list(result.shape)}")
    return {
        "input_name": input_info.name,
        "input_shape": [int(x) if isinstance(x, int) else str(x) for x in input_info.shape],
        "output_shape": [int(x) for x in result.shape],
        "output_contract": "decoded_xywh_objectness_class_probability",
        "onnx_opset": 17,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DET_FISH training requires CUDA; Cloud Run GPU was not attached")

    bucket = os.environ.get("GCS_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("GCS_BUCKET is required")
    dataset_version = os.environ.get("DETECTOR_DATASET_VERSION", "DET_DS_v0.1").strip()
    model_version = os.environ.get("DETECTOR_MODEL_VERSION", "DET_FISH_v0.1").strip()
    app_git_commit = os.environ.get("APP_GIT_COMMIT", "unknown").strip()

    work = Path("/tmp/yujian-detector")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    archive = work / f"{dataset_version}.tar.gz"
    report_path = work / "bootstrap_report.json"
    prefix = f"detector-datasets/{dataset_version}"
    download_gcs(bucket, f"{prefix}/{dataset_version}.tar.gz", archive)
    download_gcs(bucket, f"{prefix}/bootstrap_report.json", report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require_dataset(report)

    extracted = work / "dataset"
    safe_extract(archive, extracted)
    dataset_root = extracted / dataset_version
    if not (dataset_root / "annotations" / "instances_train2017.json").exists():
        raise RuntimeError(f"invalid detector dataset archive: {dataset_root}")

    pretrain = work / "yolox_nano_pretrained.pth"
    urllib.request.urlretrieve(PRETRAIN_URL, pretrain)
    if pretrain.stat().st_size < 7_000_000:
        raise RuntimeError("YOLOX-Nano pretrained checkpoint download is incomplete")

    output_root = work / "outputs"
    checkpoint = run_training(dataset_root, pretrain, output_root)
    evaluation = evaluate_checkpoint(checkpoint)
    onnx_path = work / "fish_detector_yolox_nano_v0_1.onnx"
    onnx_contract = export_onnx(checkpoint, onnx_path)

    contract_path = Path("/app/config/recognition_pipeline_v1.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint_doc = torch.load(checkpoint, map_location="cpu")
    metadata = {
        "model_version": model_version,
        "model_family": "YOLOX_NANO",
        "class_names": ["fish"],
        "dataset_version": dataset_version,
        "git_commit": app_git_commit,
        "training": {
            "epochs": int(os.environ.get("DETECTOR_EPOCHS", "30")),
            "batch_size": int(os.environ.get("DETECTOR_BATCH_SIZE", "16")),
            "ap50": evaluation["ap50"],
            "ap50_95": evaluation["ap50_95"],
            "checkpoint_best_ap50_95": float(checkpoint_doc.get("best_ap", 0.0)) if isinstance(checkpoint_doc, dict) else None,
            "pretrained_source": PRETRAIN_URL,
        },
        "dataset_report": report,
        "pipeline_contract": contract,
        "onnx": onnx_contract,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    metadata_path = work / "detector_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model_prefix = f"models/{model_version}"
    published = {
        "onnx": upload_gcs(bucket, f"{model_prefix}/{onnx_path.name}", onnx_path, "application/octet-stream"),
        "checkpoint": upload_gcs(bucket, f"{model_prefix}/best_ckpt.pth", checkpoint, "application/octet-stream"),
        "metadata": upload_gcs(bucket, f"{model_prefix}/detector_metadata.json", metadata_path, "application/json"),
        "pipeline_contract": upload_gcs(bucket, f"{model_prefix}/recognition_pipeline_v1.json", contract_path, "application/json"),
    }
    print(json.dumps({"status": "DET_FISH_TRAINING_PASS", "published": published, **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
