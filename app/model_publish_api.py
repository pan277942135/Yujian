"""One-click classifier promotion backed by the existing GitHub Actions exporter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import DatasetVersion, ModelPublishJob, ModelVersion, TrainingRun


router = APIRouter(tags=["model-publish"])

ACTIVE_STATUSES = {"CONVERTING", "VALIDATING", "PUBLISHING"}
TERMINAL_STATUSES = {"PUBLISHED", "FAILED"}
PUBLISHED_FILENAME = "fish_classifier_v0_2.tflite"
RELEASE_TAG = "mobile-model-v0.2"
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _model_prefix(artifact_uri: str) -> str:
    value = str(artifact_uri or "").strip()
    if not value.startswith("gs://") or "/" not in value[5:]:
        raise HTTPException(
            status_code=409,
            detail={"error": "MODEL_ARTIFACT_URI_INVALID", "message": "训练模型不是有效的 GCS Artifact"},
        )
    return value.rsplit("/", 1)[0]


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    body = uri.removeprefix("gs://")
    return tuple(body.split("/", 1))  # type: ignore[return-value]


def _gcs_exists(uri: str) -> bool:
    from google.cloud import storage

    bucket, object_name = _parse_gs_uri(uri)
    return bool(storage.Client().bucket(bucket).blob(object_name).exists())


def _load_gcs_json(uri: str) -> dict[str, Any] | None:
    from google.cloud import storage

    bucket, object_name = _parse_gs_uri(uri)
    blob = storage.Client().bucket(bucket).blob(object_name)
    if not blob.exists():
        return None
    value = json.loads(blob.download_as_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _github_dispatch(inputs: dict[str, str]) -> None:
    token = os.getenv("YUJIAN_GITHUB_RELEASE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_AUTH_NOT_CONFIGURED: YUJIAN_GITHUB_RELEASE_TOKEN 未配置")
    repository = os.getenv("YUJIAN_GITHUB_REPOSITORY", "pan277942135/Yujian").strip()
    workflow = os.getenv("YUJIAN_MODEL_PUBLISH_WORKFLOW", "mobile-model-analysis.yml").strip()
    ref = os.getenv("YUJIAN_MODEL_PUBLISH_REF", "main").strip()
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/workflows/{workflow}/dispatches",
        data=json.dumps({"ref": ref, "inputs": inputs}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "yujian-model-factory",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 204:
                raise RuntimeError(f"GITHUB_WORKFLOW_DISPATCH_FAILED: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        message = exc.read(1000).decode("utf-8", errors="replace")
        code = "GITHUB_AUTH_FAILED" if exc.code in {401, 403} else "GITHUB_WORKFLOW_DISPATCH_FAILED"
        raise RuntimeError(f"{code}: HTTP {exc.code} {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GITHUB_WORKFLOW_DISPATCH_FAILED: {exc.reason}") from exc


def _job_dict(job: ModelPublishJob, *, already_running: bool = False) -> dict[str, Any]:
    return {
        "ok": job.status != "FAILED",
        "already_running": already_running,
        "publish_job_id": job.publish_job_id,
        "run_id": job.run_id,
        "model_id": job.model_version,
        "model_version": job.model_version,
        "status": job.status,
        "stage": job.stage,
        "published_filename": job.published_filename,
        "target_artifact_uri": job.target_artifact_uri,
        "workflow_run_id": job.workflow_run_id,
        "workflow_run_url": job.workflow_run_url,
        "github_release_url": job.github_release_url,
        "sha256": job.sha256,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "published_at": job.published_at.isoformat() if job.published_at else None,
    }


def _latest_job(db: Session, model_version: str) -> ModelPublishJob | None:
    return db.scalar(
        select(ModelPublishJob)
        .where(ModelPublishJob.model_version == model_version)
        .order_by(ModelPublishJob.created_at.desc())
        .limit(1)
    )


def _expire_stale_jobs(db: Session) -> None:
    """Release the global lock if a workflow outlives its 35-minute timeout."""

    now = _utcnow()
    changed = False
    jobs = db.scalars(select(ModelPublishJob).where(ModelPublishJob.status.in_(ACTIVE_STATUSES))).all()
    for job in jobs:
        created_at = job.created_at
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at and (now - created_at).total_seconds() > 60 * 60:
            job.status = "FAILED"
            job.stage = "WORKFLOW_TIMEOUT"
            job.active_lock = None
            job.error_code = "MODEL_PUBLISH_TIMEOUT"
            job.error_message = "模型发布超过 60 分钟仍未结束，可重新发布"
            job.updated_at = now
            changed = True
    if changed:
        db.commit()


def _finish_published(db: Session, job: ModelPublishJob, payload: dict[str, Any]) -> None:
    now = _utcnow()
    db.execute(
        update(ModelVersion)
        .where(ModelVersion.is_production.is_(True))
        .values(is_production=False, status="CANDIDATE")
    )
    model = db.get(ModelVersion, job.model_version)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    model.is_production = True
    model.published_at = now
    model.status = "PRODUCTION"
    job.status = "PUBLISHED"
    job.stage = "COMPLETE"
    job.active_lock = None
    job.target_artifact_uri = payload.get("target_artifact_uri") or job.target_artifact_uri
    job.workflow_run_id = str(payload.get("workflow_run_id") or job.workflow_run_id or "") or None
    job.workflow_run_url = payload.get("workflow_run_url") or job.workflow_run_url
    job.github_release_url = payload.get("github_release_url") or job.github_release_url
    job.sha256 = payload.get("sha256") or job.sha256
    job.error_code = None
    job.error_message = None
    job.published_at = now
    job.updated_at = now


def _reconcile_manifest(db: Session, job: ModelPublishJob) -> None:
    if job.status not in ACTIVE_STATUSES:
        return
    try:
        manifest = _load_gcs_json(f"{job.model_prefix}/publish/publish_manifest.json")
    except Exception:
        return
    if not manifest or manifest.get("publish_job_id") != job.publish_job_id:
        return
    if manifest.get("github_release_status") == "PASS" and manifest.get("validation_status") == "PASS":
        _finish_published(db, job, manifest)
        db.commit()


def _is_classifier_pipeline(value: Any) -> bool:
    return "CLASSIFIER" in str(value or "").strip().upper()


def _supports_classifier_publish(db: Session, model: ModelVersion, run: TrainingRun) -> bool:
    """Accept a classifier when any persisted lineage field identifies it.

    Older ModelVersion rows were created before pipeline_type was copied from
    the TrainingRun and therefore carry the default WHOLE_IMAGE_V1. The source
    run, DatasetVersion, or serialized training parameters still carry the
    authoritative crop-classifier contract. Detector/whole-image rows remain
    blocked when the complete lineage has no classifier marker.
    """

    pipeline_values: list[Any] = [
        getattr(model, "pipeline_type", None),
        getattr(run, "pipeline_type", None),
    ]
    dataset_version = getattr(model, "dataset_version", None) or getattr(run, "dataset_version", None)
    if dataset_version:
        dataset = db.get(DatasetVersion, dataset_version)
        if dataset is not None:
            pipeline_values.append(getattr(dataset, "pipeline_type", None))
    try:
        params = json.loads(getattr(run, "params_json", "") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        params = {}
    if isinstance(params, dict):
        pipeline_values.append(params.get("pipeline_type"))
    return any(_is_classifier_pipeline(value) for value in pipeline_values)

@router.post("/api/models/{model_id}/publish")
def publish_model(model_id: str, request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not MODEL_ID_RE.fullmatch(model_id):
        raise HTTPException(status_code=400, detail={"error": "MODEL_ID_INVALID"})
    model = db.get(ModelVersion, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail={"error": "MODEL_NOT_FOUND"})
    run = db.get(TrainingRun, model.run_id)
    if run is None or str(run.status).upper() not in {"COMPLETED", "SUCCESS"}:
        raise HTTPException(status_code=409, detail={"error": "TRAINING_NOT_COMPLETED", "message": "训练尚未完成"})
    if not _supports_classifier_publish(db, model, run):
        raise HTTPException(status_code=409, detail={"error": "MODEL_TYPE_NOT_SUPPORTED", "message": "当前仅支持发布分类模型"})

    _expire_stale_jobs(db)
    existing = db.scalar(
        select(ModelPublishJob)
        .where(ModelPublishJob.status.in_(ACTIVE_STATUSES))
        .order_by(ModelPublishJob.created_at.desc())
        .limit(1)
    )
    if existing:
        if existing.model_version == model_id:
            return _job_dict(existing, already_running=True)
        raise HTTPException(
            status_code=409,
            detail={"error": "PUBLISH_IN_PROGRESS", "publish_job_id": existing.publish_job_id, "model_id": existing.model_version},
        )

    source_uri = str(model.artifact_uri or run.artifact_uri or "").strip()
    prefix = _model_prefix(source_uri)
    required = {
        "model_torchscript.pt": source_uri if source_uri.endswith("/model_torchscript.pt") else f"{prefix}/model_torchscript.pt",
        "class_map.json": f"{prefix}/class_map.json",
        "metrics.json": f"{prefix}/metrics.json",
    }
    try:
        missing = [name for name, uri in required.items() if not _gcs_exists(uri)]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "MODEL_ARTIFACT_CHECK_FAILED", "message": str(exc)[:500]},
        ) from exc
    if missing:
        raise HTTPException(
            status_code=409,
            detail={"error": "MODEL_ARTIFACT_NOT_FOUND", "missing": missing, "message": "缺少发布所需模型产物"},
        )

    now = _utcnow()
    callback_token = secrets.token_urlsafe(32)
    job = ModelPublishJob(
        publish_job_id=f"MP_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        run_id=run.run_id,
        model_version=model.model_version,
        source_artifact_uri=required["model_torchscript.pt"],
        model_prefix=prefix,
        target_artifact_uri=f"{prefix}/export/{model.model_version}.tflite",
        published_filename=PUBLISHED_FILENAME,
        status="CONVERTING",
        stage="WORKFLOW_DISPATCH",
        active_lock="production",
        callback_token_sha256=hashlib.sha256(callback_token.encode("utf-8")).hexdigest(),
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        active = db.scalar(select(ModelPublishJob).where(ModelPublishJob.active_lock == "production"))
        if active and active.model_version == model_id:
            return _job_dict(active, already_running=True)
        raise HTTPException(status_code=409, detail={"error": "PUBLISH_IN_PROGRESS"})

    try:
        _github_dispatch(
            {
                "model_version": model.model_version,
                "model_prefix": prefix,
                "publish_job_id": job.publish_job_id,
                "callback_url": str(request.base_url).rstrip("/") + "/api/model-publish/callback",
                "callback_token": callback_token,
            }
        )
        job.stage = "EXPORT_TFLITE"
        job.updated_at = _utcnow()
        db.commit()
        return _job_dict(job)
    except Exception as exc:
        message = str(exc)
        code = message.split(":", 1)[0] if ":" in message else "GITHUB_WORKFLOW_DISPATCH_FAILED"
        job.status = "FAILED"
        job.stage = "WORKFLOW_DISPATCH"
        job.active_lock = None
        job.error_code = code[:64]
        job.error_message = message[:2000]
        job.updated_at = _utcnow()
        db.commit()
        raise HTTPException(status_code=502, detail={"error": job.error_code, "message": job.error_message}) from exc


@router.get("/api/models/{model_id}/publish/status")
def publish_status(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    _expire_stale_jobs(db)
    job = _latest_job(db, model_id)
    if job is None:
        return {"ok": True, "model_id": model_id, "status": "NOT_PUBLISHED", "stage": None}
    _reconcile_manifest(db, job)
    return _job_dict(job)


class PublishCallback(BaseModel):
    publish_job_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=32)
    stage: str | None = Field(default=None, max_length=64)
    workflow_run_id: str | None = Field(default=None, max_length=128)
    workflow_run_url: str | None = Field(default=None, max_length=1000)
    github_release_url: str | None = Field(default=None, max_length=1000)
    target_artifact_uri: str | None = Field(default=None, max_length=2000)
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=2000)


@router.post("/api/model-publish/callback")
def publish_callback(
    payload: PublishCallback,
    x_yujian_publish_key: str = Header(default=""),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = db.get(ModelPublishJob, payload.publish_job_id)
    supplied_hash = hashlib.sha256(x_yujian_publish_key.encode("utf-8")).hexdigest()
    if job is None or not x_yujian_publish_key or not secrets.compare_digest(supplied_hash, job.callback_token_sha256):
        raise HTTPException(status_code=401, detail="发布回调认证失败")
    if job.model_version != payload.model_version:
        raise HTTPException(status_code=404, detail="发布任务不存在")
    status = payload.status.upper()
    if status not in ACTIVE_STATUSES | TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail="发布状态无效")
    if job.status in TERMINAL_STATUSES and job.status != status:
        return _job_dict(job)

    values = payload.model_dump(exclude_none=True)
    if status == "PUBLISHED":
        _finish_published(db, job, values)
    else:
        job.status = status
        job.stage = payload.stage or job.stage
        job.workflow_run_id = payload.workflow_run_id or job.workflow_run_id
        job.workflow_run_url = payload.workflow_run_url or job.workflow_run_url
        job.error_code = payload.error_code
        job.error_message = payload.error_message
        job.updated_at = _utcnow()
        if status == "FAILED":
            job.active_lock = None
    db.commit()
    return _job_dict(job)


__all__ = ["router"]
