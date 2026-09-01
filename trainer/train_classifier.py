from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from google.cloud import storage
from PIL import Image, ImageOps
from sqlalchemy.orm import Session
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.db import SessionLocal, init_db
from app.models import DatasetVersion, Evaluation, ModelVersion, TrainingRun
from app.pipeline_contract import CROP_CLASSIFIER_V1, WHOLE_IMAGE_V1, validate_pipeline_type
from evaluation.artifact_builder import build_evaluation_artifacts
from trainer.crop_dataset_validator import validate_crop_rows


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"invalid GCS URI: {uri}")
    body = uri[5:]
    if "/" not in body:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(body.split("/", 1))  # type: ignore[return-value]


def json_env(name: str, default: dict) -> dict:
    raw = os.getenv(name, "").strip()
    if not raw:
        return dict(default)
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"{name} must be a JSON object")
    return doc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FishDataset(Dataset):
    def __init__(self, rows: list[dict], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(row["local_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["class_index"])


def download_manifest(storage_client: storage.Client, uri: str) -> list[dict]:
    bucket_name, object_name = parse_gs_uri(uri)
    text = storage_client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def download_json(storage_client: storage.Client, uri: str) -> dict:
    bucket_name, object_name = parse_gs_uri(uri)
    return json.loads(storage_client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8"))


def materialize_images(storage_client: storage.Client, rows: list[dict], root: Path, pipeline_type: str = WHOLE_IMAGE_V1) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)

    def download_one(item: tuple[int, dict]) -> dict:
        idx, row = item
        # Generated crop manifests use the canonical ``gcs_uri`` field while
        # some registry exports use the more explicit ``crop_gcs_uri``.  The
        # pipeline guard in ``execute`` has already verified ``input_type=crop``
        # for this mode, so accepting both names never falls back to an
        # original-image row silently.
        if pipeline_type == CROP_CLASSIFIER_V1:
            uri = row.get("crop_gcs_uri") or row.get("gcs_uri") or row.get("local_path") or ""
        else:
            uri = row.get("gcs_uri") or row.get("crop_gcs_uri") or row.get("local_path") or ""
        uri = (uri or "").strip()
        if not uri:
            raise ValueError(f"manifest row missing gcs_uri: {row.get('image_id')}")
        if uri.startswith("gs://"):
            bucket_name, object_name = parse_gs_uri(uri)
            suffix = Path(row.get("file_name") or object_name).suffix.lower() or ".jpg"
            local_path = root / f"{idx:06d}{suffix}"
            storage_client.bucket(bucket_name).blob(object_name).download_to_filename(str(local_path))
        else:
            source_path = Path(uri)
            if not source_path.is_file():
                raise ValueError(f"manifest local_path does not exist: {source_path}")
            local_path = source_path
        out = dict(row)
        out["local_path"] = str(local_path)
        return out

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(rows)))) as pool:
        return list(pool.map(download_one, enumerate(rows)))


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        split = (row.get("split") or "").strip()
        if split not in result:
            continue
        result[split].append(row)
    return result


class CropLetterbox:
    """PIL equivalent of Android's crop → centered letterbox preprocessing."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        contained = ImageOps.contain(source, (self.size, self.size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(contained, ((self.size - contained.width) // 2, (self.size - contained.height) // 2))
        return canvas


def build_transforms(image_size: int, pipeline_type: str = WHOLE_IMAGE_V1):
    pipeline_type = validate_pipeline_type(pipeline_type)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if pipeline_type == CROP_CLASSIFIER_V1:
        transform = transforms.Compose([CropLetterbox(image_size), transforms.ToTensor(), normalize])
        return transform, transform

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_class_weights(train_rows: list[dict], num_classes: int) -> torch.Tensor:
    counts = Counter(int(row["class_index"]) for row in train_rows)
    positive = [count for count in counts.values() if count > 0]
    total = sum(positive)
    active = max(1, len(positive))
    weights = []
    for class_index in range(num_classes):
        count = counts.get(class_index, 0)
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(math.sqrt(total / (active * count)))
    return torch.tensor(weights, dtype=torch.float32)


def calculate_metrics(targets: list[int], predictions: list[int], top3_hits: int, num_classes: int) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(targets, predictions):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            confusion[truth, pred] += 1

    per_class = []
    f1_values = []
    recall_values = []
    precision_values = []
    for idx in range(num_classes):
        tp = int(confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        predicted = int(confusion[:, idx].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "class_index": idx,
                "support": support,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
        if support > 0:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

    total = len(targets)
    correct = sum(int(t == p) for t, p in zip(targets, predictions))
    return {
        "count": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "top3_accuracy": round(top3_hits / total, 6) if total else 0.0,
        "macro_precision": round(float(np.mean(precision_values)), 6) if precision_values else 0.0,
        "macro_recall": round(float(np.mean(recall_values)), 6) if recall_values else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, num_classes: int) -> dict:
    model.eval()
    losses = []
    targets: list[int] = []
    predictions: list[int] = []
    top3_hits = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            topk = logits.topk(k=min(3, num_classes), dim=1).indices
            top3_hits += int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())
            targets.extend(labels.tolist())
            predictions.extend(preds.tolist())
    metrics = calculate_metrics(targets, predictions, top3_hits, num_classes)
    metrics["loss"] = round(float(np.mean(losses)), 6) if losses else 0.0
    return metrics


def build_test_prediction_rows(
    model: nn.Module,
    test_rows: list[dict],
    class_rows: list[dict],
    image_size: int,
    batch_size: int,
    pipeline_type: str = WHOLE_IMAGE_V1,
) -> list[dict]:
    """Create provenance-rich rows for the post-training artifact contract.

    This is intentionally a read-only pass over the already evaluated test
    split.  It does not alter training weights, labels, Dataset Freeze state,
    or the existing metrics report.
    """

    if not test_rows:
        return []
    pipeline_type = validate_pipeline_type(pipeline_type)
    _train_transform, eval_transform = build_transforms(image_size, pipeline_type)
    labels = {
        int(row.get("class_index", index)): str(row.get("species_key") or row.get("common_name_en") or row.get("common_name_zh") or f"class_{index}")
        for index, row in enumerate(class_rows)
    }
    model.eval()
    result: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(test_rows), max(1, int(batch_size))):
            source_rows = test_rows[start : start + max(1, int(batch_size))]
            tensors = []
            usable_rows = []
            for source in source_rows:
                try:
                    with Image.open(source["local_path"]) as image:
                        tensors.append(eval_transform(image.convert("RGB")))
                    usable_rows.append(source)
                except Exception:
                    # The training/evaluation pass has already applied its
                    # normal data validation.  Keep this export best-effort so
                    # a single unreadable provenance row cannot rewrite it.
                    continue
            if not tensors:
                continue
            logits = model(torch.stack(tensors))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for source, confidence_value, prediction in zip(usable_rows, confidence.tolist(), predictions.tolist()):
                true_index = int(source.get("class_index", 0))
                predicted_index = int(prediction)
                row = {
                    "image_id": source.get("image_id") or source.get("id"),
                    "file_name": source.get("file_name"),
                    "true_species": source.get("species_key") or labels.get(true_index, f"class_{true_index}"),
                    "pred_species": labels.get(predicted_index, f"class_{predicted_index}"),
                    "confidence": round(float(confidence_value), 6),
                    "correct": true_index == predicted_index,
                    "local_path": source.get("local_path"),
                    "gcs_uri": source.get("gcs_uri"),
                    "scene": source.get("scene"),
                    "angle": source.get("angle"),
                    "image_quality": source.get("image_quality") or source.get("quality"),
                }
                result.append({key: value for key, value in row.items() if value not in (None, "")})
    return result


def train_model(
    rows: dict[str, list[dict]],
    class_rows: list[dict],
    params: dict,
    pipeline_type: str = WHOLE_IMAGE_V1,
) -> tuple[nn.Module, dict]:
    seed = int(params.get("seed", 20260827))
    epochs = int(params.get("epochs", 12))
    batch_size = int(params.get("batch_size", 16))
    image_size = int(params.get("image_size", 224))
    learning_rate = float(params.get("learning_rate", 0.001))
    fine_tune_learning_rate = float(params.get("fine_tune_learning_rate", learning_rate * 0.2))
    warmup_epochs = max(0, int(params.get("warmup_epochs", 2)))
    patience = max(1, int(params.get("early_stopping_patience", 4)))
    label_smoothing = float(params.get("label_smoothing", 0.05))

    set_seed(seed)
    num_classes = len(class_rows)
    if num_classes < 2:
        raise ValueError("classifier requires at least two classes")
    if not rows["train"]:
        raise ValueError("training split is empty")

    pipeline_type = validate_pipeline_type(pipeline_type)
    train_transform, eval_transform = build_transforms(image_size, pipeline_type)
    train_ds = FishDataset(rows["train"], train_transform)
    val_source = rows["val"] or rows["test"] or rows["train"]
    val_ds = FishDataset(val_source, eval_transform)
    test_source = rows["test"] or rows["val"] or rows["train"]
    test_ds = FishDataset(test_source, eval_transform)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    class_weights = make_class_weights(rows["train"], num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    if warmup_epochs > 0:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    best_state = deepcopy(model.state_dict())
    best_score = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(epochs):
        if warmup_epochs > 0 and epoch == warmup_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_learning_rate, weight_decay=0.01)

        model.train()
        epoch_losses = []
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, criterion, num_classes)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(epoch_losses)), 6) if epoch_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        score = float(val_metrics["macro_f1"])
        if score > best_score + 1e-8:
            best_score = score
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, criterion, num_classes)
    test_metrics = evaluate(model, test_loader, criterion, num_classes)

    train_counts = Counter(int(row["class_index"]) for row in rows["train"])
    val_counts = Counter(int(row["class_index"]) for row in rows["val"])
    test_counts = Counter(int(row["class_index"]) for row in rows["test"])
    warnings = []
    for class_row in class_rows:
        idx = int(class_row["class_index"])
        count = train_counts.get(idx, 0)
        if count == 0:
            warnings.append(f"{class_row.get('common_name_zh') or class_row.get('species_key')}: 训练集 0 张")
        elif count < 3:
            warnings.append(f"{class_row.get('common_name_zh') or class_row.get('species_key')}: 训练集仅 {count} 张")

    report = {
        "params": {
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "image_size": image_size,
            "learning_rate": learning_rate,
            "fine_tune_learning_rate": fine_tune_learning_rate,
            "warmup_epochs": warmup_epochs,
            "early_stopping_patience": patience,
            "label_smoothing": label_smoothing,
            "pipeline_type": pipeline_type,
        },
        "history": history,
        "split_counts": {name: len(rows[name]) for name in ("train", "val", "test")},
        "per_class_split_counts": {
            str(idx): {
                "train": train_counts.get(idx, 0),
                "val": val_counts.get(idx, 0),
                "test": test_counts.get(idx, 0),
            }
            for idx in range(num_classes)
        },
        "warnings": warnings,
        "validation": val_metrics,
        "test": test_metrics,
    }
    return model, report


def upload_text(bucket: storage.Bucket, object_name: str, text: str, content_type: str) -> str:
    bucket.blob(object_name).upload_from_string(text, content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def upload_file(bucket: storage.Bucket, object_name: str, path: Path, content_type: str) -> str:
    bucket.blob(object_name).upload_from_filename(str(path), content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def confusion_csv(class_rows: list[dict], matrix: list[list[int]]) -> str:
    buffer = io.StringIO()
    labels = [row.get("common_name_zh") or row.get("species_key") or str(row["class_index"]) for row in class_rows]
    writer = csv.writer(buffer)
    writer.writerow(["truth\\pred", *labels])
    for label, values in zip(labels, matrix):
        writer.writerow([label, *values])
    return buffer.getvalue()


def register_running(db: Session, run_id: str, dataset_version: str, model_family: str, params: dict, git_commit: str) -> TrainingRun:
    row = db.get(TrainingRun, run_id)
    if row and row.status == "COMPLETED":
        raise ValueError(f"training run already completed: {run_id}")
    if not row:
        row = TrainingRun(
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            model_family=model_family,
            params_json=json.dumps(params, ensure_ascii=False),
            seed=int(params.get("seed", 20260827)),
            status="RUNNING",
            pipeline_type=validate_pipeline_type(params.get("pipeline_type", WHOLE_IMAGE_V1)),
            detector_version=params.get("detector_version"),
            crop_version=params.get("crop_version"),
            classifier_version=params.get("classifier_version"),
        )
        db.add(row)
    else:
        row.status = "RUNNING"
        row.started_at = utcnow()
        row.finished_at = None
        row.params_json = json.dumps(params, ensure_ascii=False)
        row.pipeline_type = validate_pipeline_type(params.get("pipeline_type", WHOLE_IMAGE_V1))
        row.detector_version = params.get("detector_version")
        row.crop_version = params.get("crop_version")
        row.classifier_version = params.get("classifier_version")
    db.commit()
    return row


def execute() -> dict:
    run_id = required_env("RUN_ID")
    dataset_version = required_env("DATASET_VERSION")
    model_version = required_env("MODEL_VERSION")
    model_family = os.getenv("MODEL_FAMILY", "mobilenet_v3_small").strip() or "mobilenet_v3_small"
    if model_family != "mobilenet_v3_small":
        raise ValueError(f"unsupported model_family: {model_family}")
    params = json_env(
        "TRAINING_PARAMS_JSON",
        {
            "seed": 20260827,
            "epochs": 12,
            "batch_size": 16,
            "image_size": 224,
            "learning_rate": 0.001,
            "warmup_epochs": 2,
            "early_stopping_patience": 4,
        },
    )
    pipeline_type = validate_pipeline_type(os.getenv("PIPELINE_TYPE", params.get("pipeline_type", WHOLE_IMAGE_V1)))
    params["pipeline_type"] = pipeline_type
    if pipeline_type == CROP_CLASSIFIER_V1:
        params.setdefault("crop_version", os.getenv("CROP_VERSION", "DS_CROP_M1_v0.1"))
        params.setdefault("detector_version", os.getenv("DETECTOR_VERSION", "DET_FISH_v0.1"))
        params.setdefault("classifier_version", os.getenv("CLASSIFIER_VERSION", ""))
    params["model_version"] = model_version
    params["model_family"] = model_family
    git_commit = os.getenv("APP_GIT_COMMIT", "unknown").strip() or "unknown"

    init_db()
    storage_client = storage.Client()
    db = SessionLocal()
    try:
        dataset = db.get(DatasetVersion, dataset_version)
        if not dataset:
            raise ValueError(f"dataset not found: {dataset_version}")
        if dataset.status != "FROZEN":
            raise ValueError(f"dataset is not frozen: {dataset.status}")
        if not dataset.class_map_uri:
            raise ValueError("dataset class_map_uri is missing")

        run = register_running(db, run_id, dataset_version, model_family, params, git_commit)
        manifest = download_manifest(storage_client, dataset.manifest_uri)
        class_map_doc = download_json(storage_client, dataset.class_map_uri)
        class_rows = sorted(list(class_map_doc.get("classes") or []), key=lambda x: int(x.get("class_index", 0)))
        if not manifest:
            raise ValueError("dataset manifest is empty")
        if not class_rows:
            raise ValueError("dataset class map is empty")

        if pipeline_type == CROP_CLASSIFIER_V1:
            invalid = [row.get("image_id") for row in manifest if row.get("input_type") != "crop" or row.get("pipeline_type") != CROP_CLASSIFIER_V1]
            if invalid:
                raise ValueError(
                    "CROP_CLASSIFIER_V1 refuses original-image manifest rows; "
                    f"missing crop contract for {len(invalid)} row(s)"
                )

        with tempfile.TemporaryDirectory(prefix=f"yujian-{run_id}-") as temp_dir:
            root = Path(temp_dir)
            local_rows = materialize_images(storage_client, manifest, root / "images", pipeline_type)
            crop_validation = None
            if pipeline_type == CROP_CLASSIFIER_V1:
                crop_validation = validate_crop_rows(local_rows, require_bbox=True)
                if not crop_validation["valid"]:
                    first_error = (crop_validation.get("errors") or [{}])[0]
                    raise ValueError(
                        "CROP_DATASET_INVALID: "
                        f"{first_error.get('code', 'INVALID')}: {first_error.get('message', 'crop manifest failed validation')}"
                    )
            grouped = split_rows(local_rows)
            model, report = train_model(grouped, class_rows, params, pipeline_type=pipeline_type)
            if crop_validation is not None:
                report["crop_dataset_validation"] = crop_validation
            test_source = grouped["test"] or grouped["val"] or grouped["train"]
            prediction_rows = build_test_prediction_rows(
                model,
                test_source,
                class_rows,
                int(params.get("image_size", 224)),
                int(params.get("batch_size", 16)),
                pipeline_type=pipeline_type,
            )

            bucket_name = os.getenv("GCS_BUCKET", "").strip() or parse_gs_uri(dataset.manifest_uri)[0]
            bucket = storage_client.bucket(bucket_name)
            prefix = f"models/{model_version}/"

            report.update(
                {
                    "run_id": run_id,
                    "model_version": model_version,
                    "model_family": model_family,
                    "pipeline_type": pipeline_type,
                    "detector_version": params.get("detector_version"),
                    "crop_version": params.get("crop_version"),
                    "classifier_version": params.get("classifier_version"),
                    "dataset_version": dataset_version,
                    "dataset_manifest_uri": dataset.manifest_uri,
                    "dataset_class_map_uri": dataset.class_map_uri,
                    "git_commit": git_commit,
                    "created_at": utcnow().isoformat(),
                    "status": "COMPLETED",
                }
            )

            state_path = root / "model_state_dict.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_family": model_family,
                    "num_classes": len(class_rows),
                    "class_map": class_rows,
                    "params": report["params"],
                },
                state_path,
            )

            model.eval()
            model.cpu()
            scripted_path = root / "model_torchscript.pt"
            example = torch.randn(1, 3, int(report["params"]["image_size"]), int(report["params"]["image_size"]))
            scripted = torch.jit.trace(model, example)
            scripted.save(str(scripted_path))

            state_uri = upload_file(bucket, prefix + "model_state_dict.pt", state_path, "application/octet-stream")
            artifact_uri = upload_file(bucket, prefix + "model_torchscript.pt", scripted_path, "application/octet-stream")
            metrics_uri = upload_text(
                bucket,
                prefix + "metrics.json",
                json.dumps(report, ensure_ascii=False, indent=2),
                "application/json",
            )
            confusion_uri = upload_text(
                bucket,
                prefix + "confusion_matrix.csv",
                confusion_csv(class_rows, report["test"]["confusion_matrix"]),
                "text/csv",
            )
            evaluation_artifacts = build_evaluation_artifacts(
                report,
                prediction_rows,
                root / "evaluation_artifacts",
                model_version=model_version,
                dataset_version=dataset_version,
                class_map=class_rows,
            )
            evaluation_prefix = os.getenv("EVALUATION_ARTIFACT_PREFIX", "evaluation_artifacts").strip().strip("/") or "evaluation_artifacts"
            evaluation_root = f"{evaluation_prefix}/{model_version}/"
            evaluation_uris: dict[str, str] = {}
            artifact_content_types = {
                "metrics_path": "application/json",
                "confusion_matrix_path": "application/json",
                "predictions_path": "text/csv",
                "error_samples_path": "application/json",
                "report_path": "application/json",
            }
            for path_key, content_type in artifact_content_types.items():
                path = Path(evaluation_artifacts[path_key])
                filename = path.name
                evaluation_uris[filename] = upload_file(bucket, evaluation_root + filename, path, content_type)
            class_map_uri = upload_text(
                bucket,
                prefix + "class_map.json",
                json.dumps(class_map_doc, ensure_ascii=False, indent=2),
                "application/json",
            )
            descriptor = {
                "run_id": run_id,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "model_family": model_family,
                "pipeline_type": pipeline_type,
                "detector_version": params.get("detector_version"),
                "crop_version": params.get("crop_version"),
                "classifier_version": params.get("classifier_version"),
                "artifact_uri": artifact_uri,
                "state_dict_uri": state_uri,
                "metrics_uri": metrics_uri,
                "confusion_matrix_uri": confusion_uri,
                "class_map_uri": class_map_uri,
                "evaluation_artifact_root": f"gs://{bucket.name}/{evaluation_root}",
                "evaluation_metrics_uri": evaluation_uris["metrics.json"],
                "evaluation_confusion_matrix_uri": evaluation_uris["confusion_matrix.json"],
                "evaluation_predictions_uri": evaluation_uris["predictions.csv"],
                "evaluation_errors_uri": evaluation_uris["error_samples.json"],
                "evaluation_report_uri": evaluation_uris["report.json"],
                "git_commit": git_commit,
                "status": "COMPLETED",
                "created_at": utcnow().isoformat(),
            }
            upload_text(
                bucket,
                prefix + "model.json",
                json.dumps(descriptor, ensure_ascii=False, indent=2),
                "application/json",
            )

            existing_model = db.get(ModelVersion, model_version)
            if existing_model and existing_model.run_id != run_id:
                raise ValueError(f"model_version already belongs to another run: {model_version}")
            if not existing_model:
                existing_model = ModelVersion(
                    model_version=model_version,
                    run_id=run_id,
                    artifact_uri=artifact_uri,
                    metrics_uri=metrics_uri,
                    status="CANDIDATE",
                    notes=(
                        "YuJian CROP_CLASSIFIER_V1 MobileNetV3 Small production candidate"
                        if pipeline_type == CROP_CLASSIFIER_V1
                        else "YuJian MVP whole-image fish species classifier baseline"
                    ),
                    pipeline_type=pipeline_type,
                    detector_version=params.get("detector_version"),
                    crop_version=params.get("crop_version"),
                    classifier_version=params.get("classifier_version"),
                    dataset_version=dataset_version,
                )
                db.add(existing_model)
            else:
                existing_model.artifact_uri = artifact_uri
                existing_model.metrics_uri = metrics_uri
                existing_model.status = "CANDIDATE"
                existing_model.pipeline_type = pipeline_type
                existing_model.detector_version = params.get("detector_version")
                existing_model.crop_version = params.get("crop_version")
                existing_model.classifier_version = params.get("classifier_version")
                existing_model.dataset_version = dataset_version

            evaluation_id = f"EVAL_{model_version}_{run_id}"[:128]
            evaluation_row = db.get(Evaluation, evaluation_id)
            if not evaluation_row:
                evaluation_row = Evaluation(
                    evaluation_id=evaluation_id,
                    model_version=model_version,
                    gold_version=dataset_version,
                    metrics_uri=evaluation_uris["metrics.json"],
                    confusion_matrix_uri=evaluation_uris["confusion_matrix.json"],
                    errors_uri=evaluation_uris["error_samples.json"],
                )
                db.add(evaluation_row)
            else:
                evaluation_row.gold_version = dataset_version
                evaluation_row.metrics_uri = evaluation_uris["metrics.json"]
                evaluation_row.confusion_matrix_uri = evaluation_uris["confusion_matrix.json"]
                evaluation_row.errors_uri = evaluation_uris["error_samples.json"]

            run.status = "COMPLETED"
            run.finished_at = utcnow()
            run.artifact_uri = artifact_uri
            run.metrics_uri = metrics_uri
            db.commit()

            print(json.dumps(descriptor, ensure_ascii=False, indent=2))
            return descriptor
    except Exception as exc:
        try:
            run = db.get(TrainingRun, run_id)
            if run:
                run.status = "FAILED"
                run.finished_at = utcnow()
                bucket_name = os.getenv("GCS_BUCKET", "").strip()
                if bucket_name:
                    error_doc = {
                        "run_id": run_id,
                        "dataset_version": dataset_version,
                        "model_version": model_version,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at": utcnow().isoformat(),
                    }
                    error_uri = upload_text(
                        storage_client.bucket(bucket_name),
                        f"training-runs/{run_id}/error.json",
                        json.dumps(error_doc, ensure_ascii=False, indent=2),
                        "application/json",
                    )
                    run.metrics_uri = error_uri
                db.commit()
        except Exception:
            db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    execute()
