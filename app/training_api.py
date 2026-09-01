from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Callable

import google.auth
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import DatasetVersion, ModelVersion, TrainingRun
from app.pipeline_contract import CROP_CLASSIFIER_V1, PIPELINE_TYPES, WHOLE_IMAGE_V1, validate_pipeline_type

router = APIRouter(tags=["classifier-training"])
templates = Jinja2Templates(directory="app/templates")

DEFAULT_PROJECT_ID = "gemini-api-project-503706"
DEFAULT_REGION = "asia-east1"
DEFAULT_JOB_NAME = "yujian-classifier-trainer"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingCreate(BaseModel):
    dataset_version: str = Field(min_length=4, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    model_family: str = Field(default="mobilenet_v3_small", max_length=128)
    seed: int = 20260827
    epochs: int = Field(default=12, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=128)
    image_size: int = Field(default=224, ge=128, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    fine_tune_learning_rate: float | None = Field(default=None, gt=0, le=0.1)
    warmup_epochs: int = Field(default=2, ge=0, le=20)
    early_stopping_patience: int = Field(default=4, ge=1, le=30)
    label_smoothing: float = Field(default=0.05, ge=0, le=0.3)
    pipeline_type: str = Field(default=CROP_CLASSIFIER_V1, max_length=64)
    detector_version: str | None = Field(default=None, max_length=128)
    crop_version: str | None = Field(default=None, max_length=128)
    classifier_version: str | None = Field(default=None, max_length=128)


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    body = uri[5:]
    if "/" not in body:
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    return tuple(body.split("/", 1))  # type: ignore[return-value]


def _download_json(client: storage.Client, uri: str) -> dict:
    bucket_name, object_name = _parse_gs_uri(uri)
    text = client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8")
    return json.loads(text)


def _enrich_metrics_diagnostics(report: dict, class_map_doc: dict) -> dict:
    classes = sorted(
        list(class_map_doc.get("classes") or []),
        key=lambda row: int(row.get("class_index", 0)),
    )
    report["classes"] = [
        {
            "class_index": int(row.get("class_index", 0)),
            "species_key": row.get("species_key"),
            "common_name_zh": row.get("common_name_zh"),
            "common_name_en": row.get("common_name_en"),
        }
        for row in classes
    ]

    split_counts = report.get("per_class_split_counts") or {}
    warnings: list[dict] = []
    for row in report["classes"]:
        idx = int(row["class_index"])
        counts = split_counts.get(str(idx), split_counts.get(idx, {})) or {}
        train_count = int(counts.get("train", 0) or 0)
        val_count = int(counts.get("val", 0) or 0)
        test_count = int(counts.get("test", 0) or 0)
        name = row.get("common_name_zh") or row.get("species_key") or f"Class {idx}"

        if train_count == 0:
            warnings.append({
                "severity": "critical",
                "kind": "train_zero",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Train 0 张，无法训练该类",
            })
        elif train_count < 10:
            warnings.append({
                "severity": "warning",
                "kind": "train_low",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Train 仅 {train_count} 张，训练样本偏少",
            })

        if val_count == 0:
            warnings.append({
                "severity": "critical",
                "kind": "val_zero",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Validation 0 张，无法监控该类泛化",
            })
        elif val_count < 3:
            warnings.append({
                "severity": "warning",
                "kind": "val_low",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Validation 仅 {val_count} 张，指标波动较大",
            })

        if test_count == 0:
            warnings.append({
                "severity": "critical",
                "kind": "test_zero",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Test 0 张，最终指标没有评估该类",
            })
        elif test_count < 3:
            warnings.append({
                "severity": "warning",
                "kind": "test_low",
                "class_index": idx,
                "species": name,
                "message": f"{name}：Test 仅 {test_count} 张，最终指标可信度较低",
            })

    report["evaluation_warnings"] = warnings
    return report


def _names(payload: TrainingCreate) -> tuple[str, str]:
    suffix = payload.dataset_version.removeprefix("DS_")
    stamp = utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = (payload.run_id or f"RUN_{suffix}_{stamp}").strip()
    model_version = (payload.model_version or f"MODEL_{suffix}_{stamp}").strip()
    if not run_id.startswith("RUN_"):
        raise ValueError("训练 Run ID 必须以 RUN_ 开头")
    if not model_version.startswith("MODEL_"):
        raise ValueError("模型版本必须以 MODEL_ 开头")
    return run_id, model_version


def _params(payload: TrainingCreate, model_version: str) -> dict:
    fine_lr = payload.fine_tune_learning_rate or payload.learning_rate * 0.2
    return {
        "model_version": model_version,
        "seed": payload.seed,
        "epochs": payload.epochs,
        "batch_size": payload.batch_size,
        "image_size": payload.image_size,
        "learning_rate": payload.learning_rate,
        "fine_tune_learning_rate": fine_lr,
        "warmup_epochs": payload.warmup_epochs,
        "early_stopping_patience": payload.early_stopping_patience,
        "label_smoothing": payload.label_smoothing,
        "pipeline_type": validate_pipeline_type(payload.pipeline_type),
        "detector_version": payload.detector_version,
        "crop_version": payload.crop_version,
        "classifier_version": payload.classifier_version,
    }


def _run_cloud_job(run_id: str, dataset_version: str, model_version: str, model_family: str, params: dict) -> dict:
    project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT_ID)).strip()
    region = os.getenv("GCP_REGION", DEFAULT_REGION).strip()
    job_name = os.getenv("TRAINING_JOB_NAME", DEFAULT_JOB_NAME).strip()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/jobs/{job_name}:run"
    body = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": "RUN_ID", "value": run_id},
                        {"name": "DATASET_VERSION", "value": dataset_version},
                        {"name": "MODEL_VERSION", "value": model_version},
                        {"name": "MODEL_FAMILY", "value": model_family},
                        {"name": "PIPELINE_TYPE", "value": str(params.get("pipeline_type") or WHOLE_IMAGE_V1)},
                        {"name": "DETECTOR_VERSION", "value": str(params.get("detector_version") or "")},
                        {"name": "CROP_VERSION", "value": str(params.get("crop_version") or "")},
                        {"name": "CLASSIFIER_VERSION", "value": str(params.get("classifier_version") or "")},
                        {"name": "TRAINING_PARAMS_JSON", "value": json.dumps(params, ensure_ascii=False)},
                    ]
                }
            ],
            "taskCount": 1,
            "timeout": "3600s",
        }
    }
    response = session.post(url, json=body, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Cloud Run Job 启动失败 ({response.status_code}): {response.text[:1000]}")
    return response.json()


def run_dict(row: TrainingRun) -> dict:
    try:
        params = json.loads(row.params_json or "{}")
    except Exception:
        params = {}
    return {
        "run_id": row.run_id,
        "dataset_version": row.dataset_version,
        "git_commit": row.git_commit,
        "model_family": row.model_family,
        "params": params,
        "seed": row.seed,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "status": row.status,
        "artifact_uri": row.artifact_uri,
        "metrics_uri": row.metrics_uri,
        "pipeline_type": getattr(row, "pipeline_type", WHOLE_IMAGE_V1),
        "detector_version": getattr(row, "detector_version", None),
        "crop_version": getattr(row, "crop_version", None),
        "classifier_version": getattr(row, "classifier_version", None),
    }


def queue_training_run(
    db: Session,
    payload: TrainingCreate,
    launcher: Callable[[str, str, str, str, dict], dict] | None = None,
) -> dict:
    dataset = db.get(DatasetVersion, payload.dataset_version)
    if not dataset:
        raise ValueError("数据集不存在，请先完成 Dataset Freeze")
    if dataset.status != "FROZEN":
        raise ValueError(f"数据集尚未冻结：{dataset.status}")
    if dataset.train_count <= 0:
        raise ValueError("训练集为空，不能启动训练")
    if payload.model_family != "mobilenet_v3_small":
        raise ValueError("V0.1 仅支持 mobilenet_v3_small")
    pipeline_type = validate_pipeline_type(payload.pipeline_type)

    run_id, model_version = _names(payload)
    if db.get(TrainingRun, run_id):
        raise ValueError(f"Run ID 已存在：{run_id}")
    if db.get(ModelVersion, model_version):
        raise ValueError(f"模型版本已存在：{model_version}")

    params = _params(payload, model_version)
    row = TrainingRun(
        run_id=run_id,
        dataset_version=payload.dataset_version,
        git_commit=os.getenv("APP_GIT_COMMIT", "unknown").strip() or "unknown",
        model_family=payload.model_family,
        params_json=json.dumps(params, ensure_ascii=False),
        seed=payload.seed,
        started_at=utcnow(),
        status="QUEUED",
        pipeline_type=pipeline_type,
        detector_version=payload.detector_version,
        crop_version=payload.crop_version,
        classifier_version=payload.classifier_version,
    )
    db.add(row)
    db.commit()

    launch = launcher or _run_cloud_job
    try:
        operation = launch(run_id, payload.dataset_version, model_version, payload.model_family, params)
        params["cloud_run_operation"] = operation.get("name")
        row.params_json = json.dumps(params, ensure_ascii=False)
        db.commit()
        result = run_dict(row)
        result["model_version"] = model_version
        result["cloud_run_operation"] = operation.get("name")
        return result
    except Exception:
        row.status = "FAILED"
        row.finished_at = utcnow()
        db.commit()
        raise


@router.get("/training", response_class=HTMLResponse)
def training_page(request: Request):
    return templates.TemplateResponse(request=request, name="training.html", context={})


@router.get("/api/training/runs")
def list_training_runs(
    dataset_version: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    stmt = select(TrainingRun)
    if dataset_version:
        stmt = stmt.where(TrainingRun.dataset_version == dataset_version)
    rows = db.scalars(stmt.order_by(TrainingRun.started_at.desc()).limit(limit)).all()
    return [run_dict(row) for row in rows]


@router.get("/api/training/runs/{run_id}")
def training_run_detail(run_id: str, db: Session = Depends(get_db)):
    row = db.get(TrainingRun, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="训练 Run 不存在")
    result = run_dict(row)
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run_id))
    result["model"] = (
        {
            "model_version": model.model_version,
            "status": model.status,
            "artifact_uri": model.artifact_uri,
            "metrics_uri": model.metrics_uri,
            "pipeline_type": getattr(model, "pipeline_type", WHOLE_IMAGE_V1),
            "detector_version": getattr(model, "detector_version", None),
            "crop_version": getattr(model, "crop_version", None),
            "classifier_version": getattr(model, "classifier_version", None),
            "dataset_version": getattr(model, "dataset_version", None) or row.dataset_version,
        }
        if model
        else None
    )
    return result


@router.get("/api/training/runs/{run_id}/metrics")
def training_run_metrics(run_id: str, db: Session = Depends(get_db)):
    row = db.get(TrainingRun, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="训练 Run 不存在")
    if row.status != "COMPLETED" or not row.metrics_uri:
        raise HTTPException(status_code=409, detail="训练尚未完成，暂无指标")
    try:
        client = storage.Client()
        report = _download_json(client, row.metrics_uri)
        dataset = db.get(DatasetVersion, row.dataset_version)
        if dataset and dataset.class_map_uri:
            try:
                class_map_doc = _download_json(client, dataset.class_map_uri)
                report = _enrich_metrics_diagnostics(report, class_map_doc)
            except Exception as exc:
                report["diagnostics_error"] = f"读取 Dataset class map 失败：{exc}"
        return report
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取训练指标失败：{exc}") from exc


@router.post("/api/training/runs")
def create_training_run(payload: TrainingCreate, db: Session = Depends(get_db)):
    try:
        return queue_training_run(db, payload)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/models")
def list_models(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)).all()
    return [
        {
            "model_version": row.model_version,
            "run_id": row.run_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "artifact_uri": row.artifact_uri,
            "metrics_uri": row.metrics_uri,
            "status": row.status,
            "notes": row.notes,
            "pipeline_type": getattr(row, "pipeline_type", WHOLE_IMAGE_V1),
            "detector_version": getattr(row, "detector_version", None),
            "crop_version": getattr(row, "crop_version", None),
            "classifier_version": getattr(row, "classifier_version", None),
            "dataset_version": getattr(row, "dataset_version", None),
        }
        for row in rows
    ]
