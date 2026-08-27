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
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import DatasetVersion, ModelVersion, TrainingRun

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
        }
        if model
        else None
    )
    return result


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
        }
        for row in rows
    ]
