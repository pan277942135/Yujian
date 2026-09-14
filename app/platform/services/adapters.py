"""Read adapters for the Platform V1 console.

The adapters intentionally sit at the presentation boundary.  They compose
the existing registry models and services, hide implementation-only paths,
and return stable, Chinese-console-friendly payloads.  They do not create a
second Dataset, Training or Model domain.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.data_policy import review_group_clause
from app.fish_knowledge.api import _cover_image, load_species_with_knowledge
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.gallery import managed_knowledge_asset_url
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.species import FishSpecies
from app.main import image_dict
from app.models import (
    Batch,
    BatchCropReview,
    DatasetVersion,
    ErrorCase,
    Evaluation,
    FeedbackEvent,
    ImageAsset,
    ModelVersion,
    TrainingRun,
)
from app.dataset_models import DatasetItem
from app.platform.models import FishAsset, PipelineRun, PlatformOperationLog
from app.presence import FishPresenceResult, effective_status
from app.services.review_prefill import parse_review_signals


REVIEW_PENDING = {"pending", "needs_review", "hard_case"}
VALID_QUALITY = {"GOOD", "OK", "CLEAR", "PASS", "VALID"}
ISSUE_LABELS = {
    "low_confidence": "低置信",
    "bbox_error": "BBox 异常",
    "quality_issue": "质量异常",
}
HABITAT_LEVELS = (
    {"level": 1, "name": "小鱼缸", "capacity": 20, "resources": "基础水草与过滤", "rules": "保持单一小型鱼群"},
    {"level": 2, "name": "家庭鱼缸", "capacity": 60, "resources": "水草、躲避物与灯光", "rules": "控制鱼群密度"},
    {"level": 3, "name": "鱼塘", "capacity": 200, "resources": "浅滩、沉木与增氧", "rules": "允许多物种共存"},
    {"level": 4, "name": "湖湾", "capacity": 800, "resources": "岸线、深水区与季节", "rules": "遵循自然生态容量"},
)


def _json(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None


def _status(value: Any) -> str:
    return str(value or "UNKNOWN").strip().upper()


def _safe_count(db: Session, entity: Any, *criteria: Any) -> int:
    try:
        statement = select(func.count()).select_from(entity)
        if criteria:
            statement = statement.where(*criteria)
        return int(db.scalar(statement) or 0)
    except SQLAlchemyError:
        db.rollback()
        return 0


def _safe_scalars(db: Session, statement: Any) -> list[Any]:
    try:
        return list(db.scalars(statement).all())
    except SQLAlchemyError:
        db.rollback()
        return []


def _feedback_for_image(db: Session, image: ImageAsset) -> FeedbackEvent | None:
    conditions = [FeedbackEvent.image_gcs_uri == image.gcs_uri]
    if image.batch_id and image.image_id:
        conditions.append(
            (FeedbackEvent.materialized_batch_id == image.batch_id)
            & (FeedbackEvent.materialized_image_id == image.image_id)
        )
    try:
        return db.scalar(
            select(FeedbackEvent)
            .where(or_(*conditions))
            .order_by(FeedbackEvent.created_at.desc(), FeedbackEvent.id.desc())
        )
    except SQLAlchemyError:
        db.rollback()
        return None


def _review_item(db: Session, image: ImageAsset) -> dict[str, Any]:
    feedback = _feedback_for_image(db, image)
    signals = parse_review_signals(image.notes)
    raw = image_dict(
        image,
        db=db,
        classifier_prediction=(feedback.predicted_species if feedback else None) or signals.get("prediction"),
        classifier_confidence=(feedback.confidence if feedback else None),
    )
    confidence = _number(raw.get("ai_confidence"))
    presence = _safe_scalars(
        db,
        select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id).limit(1),
    )
    presence_row = presence[0] if presence else None
    presence_status = effective_status(presence_row) if presence_row else None
    issues: list[str] = []
    if confidence is None or confidence < 0.80:
        issues.append("low_confidence")
    if raw.get("bbox_status") != "ACCEPTED":
        issues.append("bbox_error")
    quality = _status(image.quality)
    if (quality and quality not in VALID_QUALITY) or presence_status in {"no_fish", "multi_fish", "uncertain"}:
        issues.append("quality_issue")
    return {
        "id": f"{image.batch_id}:{image.image_id}",
        "batch_id": image.batch_id,
        "image_id": image.image_id,
        "file_name": image.file_name,
        # /media is the existing controlled gateway; the registry GCS URI is
        # deliberately not returned by Platform APIs.
        "image_url": raw.get("media_url"),
        "thumbnail_url": raw.get("media_url"),
        "predicted_species": raw.get("ai_suggestion"),
        "confidence": confidence,
        "quality": image.quality,
        "presence_status": presence_status,
        "review_status": image.review_status,
        "truth_species": image.truth_species,
        "claimed_species": image.claimed_species,
        "truth_status": image.truth_status,
        "candidate_bbox": raw.get("candidate_bbox"),
        "accepted_bbox": raw.get("accepted_bbox"),
        "bbox_status": raw.get("bbox_status"),
        "bbox_review_status": raw.get("bbox_review_status"),
        "issue_types": issues,
        "issue_labels": [ISSUE_LABELS[item] for item in issues],
        "model_version": feedback.model_version if feedback else None,
        "notes": image.notes or "",
        "reviewed_at": _iso(image.reviewed_at),
    }


def _matches_issue(item: dict[str, Any], issue: str | None) -> bool:
    return not issue or issue in item.get("issue_types", [])


def _matches_status(image: ImageAsset, status: str | None) -> bool:
    if not status or status == "all":
        return True
    if status == "pending":
        return image.review_status in REVIEW_PENDING
    return image.review_status == status


def review_items(
    db: Session,
    *,
    status: str | None = None,
    batch_id: str | None = None,
    species: str | None = None,
    issue: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = 40,
) -> dict[str, Any]:
    page = max(int(page or 1), 1)
    page_size = min(max(int(page_size or 40), 1), 100)
    statement = select(ImageAsset).order_by(ImageAsset.id)
    if batch_id:
        statement = statement.where(ImageAsset.batch_id == batch_id)
    if species:
        statement = statement.where(review_group_clause(species))
    if q:
        term = f"%{q.strip()}%"
        statement = statement.where(
            or_(ImageAsset.image_id.ilike(term), ImageAsset.file_name.ilike(term), ImageAsset.source_url.ilike(term))
        )
    rows = _safe_scalars(db, statement)
    selected = []
    for image in rows:
        if not _matches_status(image, status):
            continue
        item = _review_item(db, image)
        if _matches_issue(item, issue):
            selected.append(item)
    start = (page - 1) * page_size
    return {
        "items": selected[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": len(selected),
        "has_next": start + page_size < len(selected),
    }


def review_queue(db: Session) -> dict[str, Any]:
    rows = _safe_scalars(db, select(ImageAsset).order_by(ImageAsset.id))
    counts = Counter()
    examples: dict[str, list[dict[str, Any]]] = {key: [] for key in ISSUE_LABELS}
    for image in rows:
        item = _review_item(db, image)
        for issue in item["issue_types"]:
            counts[issue] += 1
            if len(examples[issue]) < 8:
                examples[issue].append(item)
    return {
        "low_confidence": counts.get("low_confidence", 0),
        "bbox_error": counts.get("bbox_error", 0),
        "quality_issue": counts.get("quality_issue", 0),
        "total": sum(counts.values()),
        "labels": ISSUE_LABELS,
        "examples": examples,
    }


def dashboard(db: Session) -> dict[str, Any]:
    datasets = _safe_count(db, DatasetVersion)
    image_rows = _safe_count(db, ImageAsset)
    try:
        batch_image_total = int(db.scalar(select(func.coalesce(func.sum(Batch.image_count), 0))) or 0)
    except SQLAlchemyError:
        db.rollback()
        batch_image_total = 0
    images = max(image_rows, batch_image_total)
    approved_images = _safe_count(db, ImageAsset, ImageAsset.review_status == "approved")
    frozen_images = 0
    try:
        frozen_images = sum(
            _dataset_counts(db, row)["total"]
            for row in _safe_scalars(
                db,
                select(DatasetVersion).where(DatasetVersion.status.in_({"FROZEN", "READY_FOR_TRAINING"})),
            )
        )
    except Exception:
        frozen_images = 0
    valid_images = max(approved_images, frozen_images)
    models = _safe_count(db, ModelVersion)
    evaluations = _safe_count(db, Evaluation)
    pipeline_counts = {
        "running": _safe_count(db, PipelineRun, PipelineRun.status.in_({"QUEUED", "RUNNING"})),
        "success": _safe_count(db, PipelineRun, PipelineRun.status.in_({"SUCCESS", "COMPLETED"})),
        "failed": _safe_count(db, PipelineRun, PipelineRun.status.in_({"FAILED", "ERROR"})),
    }
    latest_model = None
    try:
        latest_model = db.scalar(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1))
    except SQLAlchemyError:
        db.rollback()
    queue = review_queue(db)
    asset_count = _safe_count(db, FishAsset)
    active_assets = _safe_count(db, FishAsset, FishAsset.status == "ACTIVE")
    return {
        "datasets": datasets,
        "images": images,
        "valid_images": valid_images,
        "models": models,
        "evaluations": evaluations,
        "review_queue": queue["total"],
        "current_model": latest_model.model_version if latest_model else None,
        "latest_model_version": latest_model.model_version if latest_model else None,
        "pipelines": pipeline_counts,
        "assets": asset_count,
        "asset_active": active_assets,
    }


def _dataset_counts(db: Session, dataset: DatasetVersion) -> dict[str, int]:
    total = int(dataset.train_count or 0) + int(dataset.val_count or 0) + int(dataset.test_count or 0)
    item_count = _safe_count(db, DatasetItem, DatasetItem.dataset_version == dataset.dataset_version)
    if item_count:
        total = item_count
    pending = 0
    approved = 0
    try:
        pending = int(
            db.scalar(
                select(func.count())
                .select_from(DatasetItem)
                .join(ImageAsset, DatasetItem.image_asset_id == ImageAsset.id)
                .where(
                    DatasetItem.dataset_version == dataset.dataset_version,
                    ImageAsset.review_status.in_(REVIEW_PENDING),
                )
            )
            or 0
        )
        approved = int(
            db.scalar(
                select(func.count())
                .select_from(DatasetItem)
                .join(ImageAsset, DatasetItem.image_asset_id == ImageAsset.id)
                .where(DatasetItem.dataset_version == dataset.dataset_version, ImageAsset.review_status == "approved")
            )
            or 0
        )
    except SQLAlchemyError:
        db.rollback()
    valid = approved or (total if _status(dataset.status) in {"FROZEN", "READY_FOR_TRAINING"} else 0)
    return {
        "total": total,
        "valid": valid,
        "pending": pending,
        "training": int(dataset.train_count or 0),
    }


def datasets(db: Session) -> list[dict[str, Any]]:
    rows = _safe_scalars(db, select(DatasetVersion).order_by(DatasetVersion.created_at.desc()))
    result = []
    for row in rows:
        counts = _dataset_counts(db, row)
        result.append(
            {
                "id": row.dataset_version,
                "name": row.dataset_version,
                "total_images": counts["total"],
                "valid_images": counts["valid"],
                "pending_review": counts["pending"],
                "train_ready": counts["training"],
                "status": _status(row.status),
                "pipeline_type": getattr(row, "pipeline_type", None),
                "created_at": _iso(row.created_at),
            }
        )
    known = {item["id"] for item in result}
    for batch in _safe_scalars(db, select(Batch).order_by(Batch.created_at.desc())):
        if batch.batch_id in known:
            continue
        total = int(batch.image_count or 0) or _safe_count(db, ImageAsset, ImageAsset.batch_id == batch.batch_id)
        valid = _safe_count(db, ImageAsset, ImageAsset.batch_id == batch.batch_id, ImageAsset.review_status == "approved")
        pending = _safe_count(db, ImageAsset, ImageAsset.batch_id == batch.batch_id, ImageAsset.review_status.in_(REVIEW_PENDING))
        result.append(
            {
                "id": batch.batch_id,
                "name": batch.batch_id,
                "total_images": total,
                "valid_images": valid,
                "pending_review": pending,
                "train_ready": 0,
                "status": _status(batch.status),
                "pipeline_type": "BATCH_INGEST",
                "created_at": _iso(batch.created_at),
            }
        )
    return result


def dataset_detail(db: Session, dataset_id: str) -> dict[str, Any] | None:
    row = db.get(DatasetVersion, dataset_id)
    if row:
        counts = _dataset_counts(db, row)
        metadata = _json(getattr(row, "metadata_json", None), {}) or {}
        status = _status(row.status)
        pipeline_type = getattr(row, "pipeline_type", None)
        created_at = _iso(row.created_at)
    else:
        batch = db.get(Batch, dataset_id)
        if not batch:
            return None
        total = int(batch.image_count or 0) or _safe_count(db, ImageAsset, ImageAsset.batch_id == dataset_id)
        valid = _safe_count(db, ImageAsset, ImageAsset.batch_id == dataset_id, ImageAsset.review_status == "approved")
        pending = _safe_count(db, ImageAsset, ImageAsset.batch_id == dataset_id, ImageAsset.review_status.in_(REVIEW_PENDING))
        counts = {"total": total, "valid": valid, "pending": pending, "training": 0}
        metadata = {}
        status = _status(batch.status)
        pipeline_type = "BATCH_INGEST"
        created_at = _iso(batch.created_at)
    return {
        "id": dataset_id,
        "name": dataset_id,
        "status": status,
        "pipeline_type": pipeline_type,
        "created_at": created_at,
        "counts": {
            "uploaded": counts["total"],
            "valid": counts["valid"],
            "ai_filtered": max(counts["total"] - counts["valid"], 0),
            "pending_review": counts["pending"],
            "human_confirmed": counts["valid"] - counts["pending"],
            "train_ready": counts["training"],
        },
        "cleaning": metadata.get("clean_report", metadata.get("cleaning", {})) if isinstance(metadata, dict) else {},
    }


def clean_report(db: Session, dataset_id: str) -> dict[str, Any] | None:
    detail = dataset_detail(db, dataset_id)
    if detail is None:
        return None
    stored = detail.get("cleaning") if isinstance(detail.get("cleaning"), dict) else {}
    total = int(detail["counts"]["uploaded"] or 0)
    invalid = {
        "blur": int(stored.get("blur", stored.get("blurred", 0)) or 0),
        "duplicate": int(stored.get("duplicate", stored.get("duplicates", 0)) or 0),
        "no_fish": int(stored.get("no_fish", 0) or 0),
        "multi_fish": int(stored.get("multi_fish", 0) or 0),
        "scene": int(stored.get("scene", stored.get("non_target_scene", 0)) or 0),
    }
    if not any(invalid.values()):
        try:
            items = _safe_scalars(
                db,
                select(DatasetItem).where(DatasetItem.dataset_version == dataset_id),
            )
            image_ids = [item.image_asset_id for item in items]
            if image_ids:
                images = _safe_scalars(db, select(ImageAsset).where(ImageAsset.id.in_(image_ids)))
                for image in images:
                    if _status(image.quality) in {"BLUR", "BLURRED"}:
                        invalid["blur"] += 1
                presence_rows = _safe_scalars(
                    db,
                    select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids)),
                )
                for row in presence_rows:
                    state = effective_status(row)
                    if state in {"no_fish", "multi_fish"}:
                        invalid[state] += 1
        except SQLAlchemyError:
            db.rollback()
    invalid_total = min(sum(invalid.values()), total)
    valid = max(total - invalid_total, int(detail["counts"].get("valid", 0) or 0))
    return {
        "dataset_id": dataset_id,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "invalid_total": invalid_total,
        "valid_ratio": round(valid / total, 4) if total else 0.0,
    }


def training_jobs(db: Session) -> list[dict[str, Any]]:
    rows = _safe_scalars(db, select(TrainingRun).order_by(TrainingRun.started_at.desc(), TrainingRun.run_id.desc()))
    result = []
    for row in rows:
        params = _json(row.params_json, {}) or {}
        result.append(
            {
                "id": row.run_id,
                "task_id": row.run_id,
                "dataset": row.dataset_version,
                "model": row.model_family,
                "model_type": "classifier" if "CLASSIFIER" in _status(getattr(row, "pipeline_type", "")) else "detector",
                "status": _status(row.status),
                "started_at": _iso(row.started_at),
                "finished_at": _iso(row.finished_at),
                "progress": params.get("progress"),
                "result": _json(params.get("metrics"), {}) or {},
            }
        )
    return result


def _load_json_uri(uri: str | None) -> dict[str, Any]:
    if not uri:
        return {}
    value = str(uri).strip()
    try:
        if value.startswith("gs://"):
            from app.frozen_crop_bridge import _read_uri

            raw, _ = _read_uri(value)
            parsed = json.loads(raw.decode("utf-8"))
        else:
            parsed = json.loads(Path(value).read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {}


def _metric_value(document: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = document.get(key)
        if value is None and isinstance(document.get("metrics"), dict):
            value = document["metrics"].get(key)
        number = _number(value)
        if number is not None:
            if 1.0 < number <= 100.0:
                number /= 100.0
            return number
    return None


def _metrics(document: dict[str, Any]) -> dict[str, float | None]:
    return {
        "accuracy": _metric_value(document, "accuracy", "acc", "top1", "top1_accuracy"),
        "precision": _metric_value(document, "precision"),
        "recall": _metric_value(document, "recall"),
        "f1": _metric_value(document, "f1", "f1_score"),
        "top1": _metric_value(document, "top1", "top1_accuracy", "accuracy", "acc"),
        "top3": _metric_value(document, "top3", "top3_accuracy"),
    }


def _evaluation_for_model(db: Session, model: ModelVersion) -> Evaluation | None:
    return db.scalar(
        select(Evaluation)
        .where(Evaluation.model_version == model.model_version)
        .order_by(Evaluation.created_at.desc(), Evaluation.evaluation_id.desc())
        .limit(1)
    )


def models(db: Session) -> list[dict[str, Any]]:
    rows = _safe_scalars(db, select(ModelVersion).order_by(ModelVersion.created_at.desc()))
    result = []
    for row in rows:
        evaluation = _evaluation_for_model(db, row)
        document = _load_json_uri(evaluation.metrics_uri if evaluation else row.metrics_uri)
        metric = _metrics(document)
        model_type = "分类模型" if "CLASSIFIER" in _status(getattr(row, "pipeline_type", "")) else "检测模型"
        result.append(
            {
                "id": row.model_version,
                "name": "FishClassifier" if model_type == "分类模型" else "FishDetector",
                "version": row.model_version,
                "type": model_type,
                "metrics": metric,
                "status": _status(row.status),
                "status_label": _model_status_label(row.status),
                "dataset": getattr(row, "dataset_version", None),
                "created_at": _iso(row.created_at),
                "evaluation_id": evaluation.evaluation_id if evaluation else None,
            }
        )
    return result


def _model_status_label(value: Any) -> str:
    return {
        "TRAINED": "训练完成",
        "STAGING": "测试中",
        "PRODUCTION": "线上",
        "DEPRECATED": "废弃",
        "SUCCESS": "训练完成",
    }.get(_status(value), str(value or "未知"))


def evaluation(db: Session, model_id: str) -> dict[str, Any] | None:
    model = db.get(ModelVersion, model_id)
    if not model:
        return None
    row = _evaluation_for_model(db, model)
    document = _load_json_uri(row.metrics_uri if row else model.metrics_uri)
    metric = _metrics(document)
    matrix = document.get("confusion_matrix", document.get("confusionMatrix", []))
    errors = document.get("errors", [])
    if row and row.errors_uri and not errors:
        loaded_errors = _load_json_uri(row.errors_uri)
        errors = loaded_errors.get("errors", []) if isinstance(loaded_errors, dict) else []
    if row:
        db_errors = _safe_scalars(
            db,
            select(ErrorCase).where(ErrorCase.evaluation_id == row.evaluation_id).order_by(ErrorCase.created_at.desc()).limit(50),
        )
        if db_errors:
            errors = [
                {
                    "image_id": item.image_id,
                    "truth": item.truth_species,
                    "prediction": item.predicted_species,
                    "confidence": item.confidence,
                }
                for item in db_errors
            ]
    return {
        "model_id": model.model_version,
        "model_version": model.model_version,
        "evaluation_id": row.evaluation_id if row else None,
        "metrics": metric,
        "confusion_matrix": matrix if isinstance(matrix, list) else [],
        "errors": errors if isinstance(errors, list) else [],
        "available": bool(document or row),
    }


def _category_changes(current_doc: dict[str, Any], previous_doc: dict[str, Any]) -> list[dict[str, Any]]:
    def per_class(document: dict[str, Any]) -> dict[str, float]:
        source = document.get("per_class") or document.get("per_class_metrics") or document.get("class_metrics") or {}
        result: dict[str, float] = {}
        if not isinstance(source, dict):
            return result
        for key, value in source.items():
            if isinstance(value, dict):
                value = value.get("f1", value.get("accuracy", value.get("recall")))
            number = _number(value)
            if number is not None:
                if 1.0 < number <= 100.0:
                    number /= 100.0
                result[str(key)] = number
        return result

    current = per_class(current_doc)
    previous = per_class(previous_doc)
    result = []
    for species in sorted(set(current) | set(previous)):
        now, before = current.get(species), previous.get(species)
        if now is None or before is None:
            continue
        result.append({"species": species, "previous": before, "current": now, "delta": round(now - before, 4)})
    return sorted(result, key=lambda item: item["delta"], reverse=True)


def training_report(db: Session, run_id: str) -> dict[str, Any] | None:
    run = db.get(TrainingRun, run_id)
    if not run:
        return None
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run_id))
    current_doc = _load_json_uri(model.metrics_uri if model else run.metrics_uri)
    previous = None
    if model:
        previous = db.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.created_at < model.created_at,
                ModelVersion.pipeline_type == getattr(model, "pipeline_type", None),
            )
            .order_by(ModelVersion.created_at.desc())
            .limit(1)
        )
    previous_doc = _load_json_uri(previous.metrics_uri if previous else None)
    current_metric = _metrics(current_doc)
    previous_metric = _metrics(previous_doc)
    recommendations = current_doc.get("recommendations") if isinstance(current_doc.get("recommendations"), list) else []
    if not recommendations:
        category_changes = _category_changes(current_doc, previous_doc)
        decreases = [row for row in category_changes if row["delta"] < 0]
        recommendations = [f"增加{row['species']}样本，重点覆盖容易混淆的场景" for row in decreases[:3]]
    else:
        category_changes = _category_changes(current_doc, previous_doc)
    if not recommendations and not current_doc:
        recommendations = ["完成评估后自动生成下一轮采集建议"]
    return {
        "run_id": run.run_id,
        "status": _status(run.status),
        "model_version": model.model_version if model else None,
        "current": current_metric,
        "previous": previous_metric,
        "previous_model_version": previous.model_version if previous else None,
        "category_changes": category_changes,
        "recommendations": recommendations,
        "metrics_available": bool(current_doc),
    }


def _pipeline_stages(row: PipelineRun) -> list[dict[str, Any]]:
    raw = _json(row.stage_json, {})
    if isinstance(raw, dict):
        raw = raw.get("stages", [])
    if not isinstance(raw, list):
        return []
    result = []
    for stage in raw:
        if not isinstance(stage, dict):
            continue
        result.append(
            {
                "name": stage.get("name") or stage.get("stage") or "未知阶段",
                "status": _status(stage.get("status")),
                "duration_ms": stage.get("duration_ms"),
                "model_version": stage.get("model_version"),
                "error": stage.get("error") or stage.get("error_message"),
            }
        )
    return result


def _pipeline_item(row: PipelineRun) -> dict[str, Any]:
    return {
        "id": row.run_id,
        "task_id": row.run_id,
        "type": row.pipeline_type,
        "status": _status(row.status),
        "current_stage": row.current_stage,
        "duration_ms": row.duration_ms,
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "stages": _pipeline_stages(row),
        "error_stage": row.error_stage,
        "error_message": row.error_message,
        "source": {"batch_id": row.source_batch_id, "image_id": row.source_image_id},
    }


def pipelines(db: Session) -> list[dict[str, Any]]:
    rows = _safe_scalars(db, select(PipelineRun).order_by(PipelineRun.created_at.desc(), PipelineRun.run_id.desc()))
    return [_pipeline_item(row) for row in rows]


def pipeline_detail(db: Session, pipeline_id: str) -> dict[str, Any] | None:
    row = db.get(PipelineRun, pipeline_id)
    return _pipeline_item(row) if row else None


def _asset_url(asset_id: str, kind: str, uri: str | None) -> str | None:
    return f"/api/platform/assets/{asset_id}/media/{kind}" if str(uri or "").strip() else None


def _asset_item(row: FishAsset) -> dict[str, Any]:
    return {
        "id": row.asset_id,
        "asset_id": row.asset_id,
        "species": row.species,
        "source": {"batch_id": row.source_batch_id, "image_id": row.source_image_id},
        "status": _status(row.status),
        "version": row.version,
        "created_at": _iso(row.created_at),
        "pipeline_run_id": row.pipeline_run_id,
        "has_original": bool(row.original_uri),
        "has_mask": bool(row.mask_uri),
        "has_transparent": bool(row.transparent_uri),
        "has_sticker": bool(row.sticker_uri),
        "original_url": _asset_url(row.asset_id, "original", row.original_uri),
        "mask_url": _asset_url(row.asset_id, "mask", row.mask_uri),
        "transparent_url": _asset_url(row.asset_id, "transparent", row.transparent_uri),
        "sticker_url": _asset_url(row.asset_id, "sticker", row.sticker_uri),
    }


def assets(db: Session) -> list[dict[str, Any]]:
    rows = _safe_scalars(db, select(FishAsset).order_by(FishAsset.created_at.desc(), FishAsset.asset_id.desc()))
    return [_asset_item(row) for row in rows]


def asset_detail(db: Session, asset_id: str) -> dict[str, Any] | None:
    row = db.get(FishAsset, asset_id)
    return _asset_item(row) if row else None


def asset_uri(db: Session, asset_id: str, kind: str) -> str | None:
    row = db.get(FishAsset, asset_id)
    if not row:
        return None
    return {
        "original": row.original_uri,
        "mask": row.mask_uri,
        "transparent": row.transparent_uri,
        "sticker": row.sticker_uri,
    }.get(kind)


def knowledge(db: Session) -> list[dict[str, Any]]:
    statement = (
        select(FishSpecies)
        .options(selectinload(FishSpecies.gallery), selectinload(FishSpecies.cover), selectinload(FishSpecies.cards), selectinload(FishSpecies.profile), selectinload(FishSpecies.fishing))
        .order_by(FishSpecies.name_cn)
    )
    rows = _safe_scalars(db, statement)
    result = []
    for row in rows:
        active_cards = [card for card in row.cards if card.status == "ACTIVE"]
        result.append(
            {
                "id": row.id,
                "species": row.name_cn,
                "status": _status(row.status),
                "images": len(row.gallery) + (1 if row.cover else 0) + len(active_cards),
                "cover_image": _cover_image(row),
                "summary": row.summary,
                "profile_ready": bool(row.profile),
                "fishing_ready": bool(row.fishing),
                "cards_ready": len(active_cards),
            }
        )
    return result


def knowledge_detail(db: Session, species_id: str) -> dict[str, Any] | None:
    row = load_species_with_knowledge(db, species_id, active_only=False)
    if not row:
        return None
    return {
        "id": row.id,
        "species": row.name_cn,
        "status": _status(row.status),
        "tabs": {
            "基础信息": {"summary": row.summary, "scientific_name": row.scientific_name, "category": row.category},
            "识别特征": {"features": list((row.profile.features if row.profile else []) or [])},
            "钓法": {
                "water_layer": row.fishing.water_layer if row.fishing else None,
                "bait": list((row.fishing.bait if row.fishing else []) or []),
                "method": list((row.fishing.method if row.fishing else []) or []),
            },
            "相似鱼": [],
        },
        "gallery": [item.url for item in row.gallery],
        "cover_image": _cover_image(row),
    }


def habitat() -> list[dict[str, Any]]:
    return [dict(item) for item in HABITAT_LEVELS]


def tasks(db: Session) -> list[dict[str, Any]]:
    result = []
    for item in training_jobs(db):
        result.append({"id": item["id"], "type": "训练任务", "status": item["status"], "duration_ms": None, "created_at": item["started_at"]})
    for item in pipelines(db):
        result.append({"id": item["id"], "type": "Pipeline 任务", "status": item["status"], "duration_ms": item["duration_ms"], "created_at": item["started_at"]})
    for item in assets(db):
        result.append({"id": item["id"], "type": "资产任务", "status": item["status"], "duration_ms": None, "created_at": item["created_at"]})
    return result


def logs(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    rows = _safe_scalars(
        db,
        select(PlatformOperationLog).order_by(PlatformOperationLog.created_at.desc(), PlatformOperationLog.id.desc()).limit(min(max(limit, 1), 200)),
    )
    return [
        {
            "id": row.id,
            "operation": row.operation_type,
            "resource_type": row.resource_type,
            "resource_id": row.resource_id,
            "status": _status(row.status),
            "message": row.message,
            "actor": row.actor,
            "created_at": _iso(row.created_at),
        }
        for row in rows
    ]


def record_operation(
    db: Session,
    operation_type: str,
    resource_type: str,
    resource_id: str | None,
    *,
    status: str = "SUCCESS",
    message: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    db.add(
        PlatformOperationLog(
            operation_type=operation_type,
            resource_type=resource_type,
            resource_id=resource_id,
            status=status,
            message=message,
            detail_json=json.dumps(detail or {}, ensure_ascii=False),
            actor="platform",
        )
    )


__all__ = [
    "ISSUE_LABELS",
    "asset_detail",
    "asset_uri",
    "assets",
    "clean_report",
    "dashboard",
    "dataset_detail",
    "datasets",
    "evaluation",
    "habitat",
    "knowledge",
    "knowledge_detail",
    "logs",
    "models",
    "pipeline_detail",
    "pipelines",
    "record_operation",
    "review_items",
    "review_queue",
    "tasks",
    "training_jobs",
    "training_report",
]
