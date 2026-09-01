from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import get_db
from app.models import DatasetVersion, Evaluation, ImageAsset, ModelVersion, SpeciesCatalog
from app.intelligence.confusion_analyzer import build_confusion_report
from app.intelligence.data_gap_analyzer import analyze_data_gaps
from app.intelligence.hard_case_miner import mine_hard_cases
from app.intelligence.task_generator import generate_collection_task, write_collection_task
from app.intelligence.detector_error_analyzer import analyze_detector_errors

router = APIRouter(tags=["model-intelligence"])
templates = Jinja2Templates(directory="app/templates")

DEFAULT_MODEL_VERSION = "MODEL_M1_v0.3"
DEFAULT_EVALUATION_PATHS = (
    "evaluation_artifacts/{model_version}/metrics.json",
    "var/evaluation_artifacts/{model_version}/metrics.json",
    "var/intelligence/{model_version}/evaluation.json",
    "var/intelligence/{model_version}/metrics.json",
    "evaluations/{model_version}/metrics.json",
    "models/{model_version}/metrics.json",
)


class IntelligenceTaskRequest(BaseModel):
    model_version: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)
    sequence: int = Field(default=1, ge=1, le=9999)
    scenes: list[str] | None = None


class IntelligenceAnalyzeRequest(BaseModel):
    model_version: str | None = Field(default=None, max_length=128)
    evaluation_path: str | None = Field(default=None, max_length=2048)
    target_config_path: str | None = Field(default=None, max_length=2048)
    output_root: str = Field(default="var/intelligence", max_length=2048)
    mine_hard_cases: bool = False
    strict_hard_cases: bool = False


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _read_json_source(source: str | Path, storage_client: Any = None) -> Any:
    source = str(source)
    if source.startswith("gs://"):
        bucket_name, object_name = _parse_gs_uri(source)
        client = storage_client or storage.Client()
        return json.loads(client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8"))
    path = Path(source)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return _parse_csv_artifact(handle.read())
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _sibling_source(source: str | Path, filename: str) -> str:
    """Return a sibling artifact URI/path without changing its storage root."""

    value = str(source)
    if value.startswith("gs://"):
        bucket_name, object_name = _parse_gs_uri(value)
        parent = object_name.rsplit("/", 1)[0] if "/" in object_name else ""
        object_path = f"{parent}/{filename}" if parent else filename
        return f"gs://{bucket_name}/{object_path}"
    path = Path(value)
    return str(path.parent / filename)


def _merge_evaluation_artifacts(
    document: Any,
    source: str | Path,
    storage_client: Any = None,
) -> Any:
    """Join the five v1 files into the legacy evaluation shape used by Intelligence.

    The training worker writes each artifact independently so an interrupted
    upload cannot replace an older immutable artifact.  Reads are therefore
    best-effort and retain whichever files are present.
    """

    merged = dict(document) if isinstance(document, Mapping) else {}
    merged.setdefault("source_uri", str(source))
    siblings = {
        "confusion_matrix": "confusion_matrix.json",
        "predictions": "predictions.csv",
        "errors": "error_samples.json",
        "report": "report.json",
    }
    for kind, filename in siblings.items():
        try:
            value = _read_uri_or_path(_sibling_source(source, filename), storage_client)
        except Exception:
            continue
        if kind == "confusion_matrix" and isinstance(value, Mapping):
            if value.get("labels") is not None:
                merged["labels"] = value.get("labels")
            if value.get("matrix") is not None:
                merged["confusion_matrix"] = value.get("matrix")
        elif kind == "predictions":
            rows = value.get("samples") if isinstance(value, Mapping) else value
            if isinstance(rows, list):
                merged["samples"] = rows
        elif kind == "errors":
            rows = value.get("samples") if isinstance(value, Mapping) else value
            if isinstance(rows, list):
                merged["errors"] = rows
        elif kind == "report" and isinstance(value, Mapping):
            merged["evaluation_artifact_report"] = value
    return merged


def _parse_csv_artifact(text: str) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return {"samples": []}
    header = rows[0]
    if header and header[0].strip().lower() in {"truth\\pred", "truth/pred", "true\\pred"} and len(header) > 1:
        labels = [str(value).strip() for value in header[1:]]
        matrix: list[list[int]] = []
        for row in rows[1:]:
            values: list[int] = []
            for value in row[1 : len(labels) + 1]:
                try:
                    values.append(int(value or 0))
                except (TypeError, ValueError):
                    values.append(0)
            values.extend([0] * (len(labels) - len(values)))
            matrix.append(values)
        return {"labels": labels, "confusion_matrix": matrix}
    return {"samples": list(csv.DictReader(io.StringIO(text)))}


def _read_uri_or_path(source: str | Path, storage_client: Any = None) -> Any:
    try:
        return _read_json_source(source, storage_client)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        source_text = str(source)
        if source_text.startswith("gs://") and source_text.lower().endswith(".csv"):
            bucket_name, object_name = _parse_gs_uri(source_text)
            client = storage_client or storage.Client()
            text = client.bucket(bucket_name).blob(object_name).download_as_text(encoding="utf-8-sig")
            return _parse_csv_artifact(text)
        raise


def _latest_model_version(db: Session, requested: str | None = None) -> str:
    if requested and requested.strip():
        return requested.strip()
    env_version = os.getenv("INTELLIGENCE_MODEL_VERSION", "").strip()
    if env_version:
        return env_version
    latest_evaluation = db.scalar(select(Evaluation).order_by(Evaluation.created_at.desc()).limit(1))
    if latest_evaluation:
        return latest_evaluation.model_version
    latest_model = db.scalar(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1))
    if latest_model:
        return latest_model.model_version
    return DEFAULT_MODEL_VERSION


def _candidate_artifacts(db: Session, model_version: str) -> list[str]:
    candidates: list[str] = []
    artifact_root = os.getenv("EVALUATION_ARTIFACT_ROOT", os.getenv("INTELLIGENCE_ARTIFACT_ROOT", "")).strip()
    if artifact_root:
        candidates.append(f"{artifact_root.rstrip('/')}/{model_version}/metrics.json" if "{model_version}" not in artifact_root else artifact_root.format(model_version=model_version).rstrip("/") + "/metrics.json")
    bucket_name = os.getenv("GCS_BUCKET", os.getenv("YUJIAN_GCS_BUCKET", "")).strip()
    if bucket_name:
        candidates.append(f"gs://{bucket_name}/evaluation_artifacts/{model_version}/metrics.json")
    env_path = os.getenv("INTELLIGENCE_EVALUATION_PATH", os.getenv("MODEL_INTELLIGENCE_EVALUATION_PATH", "")).strip()
    if env_path:
        candidates.append(env_path.format(model_version=model_version))
    evaluation = db.scalar(
        select(Evaluation)
        .where(Evaluation.model_version == model_version)
        .order_by(Evaluation.created_at.desc())
        .limit(1)
    )
    if evaluation:
        for uri in (evaluation.metrics_uri, evaluation.confusion_matrix_uri, evaluation.errors_uri):
            if uri:
                candidates.append(uri)
    model = db.get(ModelVersion, model_version)
    if model and model.metrics_uri:
        candidates.append(model.metrics_uri)
    candidates.extend(path.format(model_version=model_version) for path in DEFAULT_EVALUATION_PATHS)
    return list(dict.fromkeys(candidates))


def _merge_error_samples(document: Any, db: Session, model_version: str, storage_client: Any = None) -> Any:
    if not isinstance(document, Mapping):
        return document
    if document.get("samples") or document.get("evaluation_samples") or document.get("predictions"):
        return document
    evaluation = db.scalar(
        select(Evaluation)
        .where(Evaluation.model_version == model_version)
        .order_by(Evaluation.created_at.desc())
        .limit(1)
    )
    if not evaluation or not evaluation.errors_uri:
        return document
    try:
        errors = _read_uri_or_path(evaluation.errors_uri, storage_client)
    except Exception:
        return document
    if isinstance(errors, Mapping) and errors.get("samples"):
        merged = dict(document)
        merged["samples"] = errors["samples"]
        return merged
    return document


def load_evaluation_document(db: Session, model_version: str, *, storage_client: Any = None) -> tuple[Any | None, str | None, str | None]:
    """Resolve the latest existing evaluation artifact without inventing data."""

    errors: list[str] = []
    for candidate in _candidate_artifacts(db, model_version):
        try:
            document = _read_uri_or_path(candidate, storage_client)
            if isinstance(document, Mapping):
                document = dict(document)
                document.setdefault("model_version", model_version)
                document = _merge_evaluation_artifacts(document, candidate, storage_client)
                return document, candidate, None
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    reason = "未找到评估产物；请先上传 metrics.json 或配置 INTELLIGENCE_EVALUATION_PATH。"
    if errors and os.getenv("INTELLIGENCE_DEBUG", "").strip() == "1":
        reason += " " + errors[-1]
    return None, None, reason


def _registry_manifest_rows(db: Session) -> list[dict[str, Any]]:
    catalog = {
        row.common_name_zh: row.species_key
        for row in db.scalars(select(SpeciesCatalog)).all()
        if row.common_name_zh and row.species_key
    }
    rows: list[dict[str, Any]] = []
    for image in db.scalars(select(ImageAsset).order_by(ImageAsset.id)).all():
        claimed = (image.claimed_species or "").strip()
        truth = (image.truth_species or "").strip()
        name = truth or claimed
        species_key = catalog.get(name, name)
        if not species_key:
            continue
        rows.append(
            {
                "image_id": image.image_id,
                "species_key": species_key,
                "claimed_species": claimed,
                "truth_species": truth,
                "scene": image.scene,
                "image_quality": image.quality,
                "quality": image.quality,
                "review_status": image.review_status,
                "batch_id": image.batch_id,
            }
        )
    return rows


def _load_class_map(db: Session, model_version: str, document: Any, storage_client: Any = None) -> Any:
    if not isinstance(document, Mapping) or document.get("classes"):
        return document
    model = db.get(ModelVersion, model_version)
    dataset = None
    if model:
        # ModelVersion references a run only through the database relation in
        # the existing schema; resolve it with a small explicit query to avoid
        # changing that schema.
        from app.models import TrainingRun

        run = db.get(TrainingRun, model.run_id)
        if run:
            dataset = db.get(DatasetVersion, run.dataset_version)
    if not dataset or not dataset.class_map_uri:
        return document
    try:
        class_map = _read_uri_or_path(dataset.class_map_uri, storage_client)
    except Exception:
        return document
    if isinstance(class_map, Mapping) and class_map.get("classes"):
        merged = dict(document)
        merged["classes"] = class_map["classes"]
        return merged
    return document


def _metrics_for_dashboard(document: Any) -> dict[str, Any]:
    """Expose the canonical metrics envelope without changing legacy reports."""

    if not isinstance(document, Mapping):
        return {}
    raw = document.get("metrics")
    if not isinstance(raw, Mapping):
        raw = document.get("test")
    if not isinstance(raw, Mapping):
        return {}
    result = dict(raw)
    if document.get("test_samples") not in (None, ""):
        result["test_samples"] = document.get("test_samples")
    elif result.get("count") not in (None, ""):
        result["test_samples"] = result.get("count")
    return result


def build_intelligence_payload(
    db: Session,
    *,
    model_version: str | None = None,
    target_config: Any = None,
    evaluation_document: Any = None,
    storage_client: Any = None,
) -> dict[str, Any]:
    resolved_model = _latest_model_version(db, model_version)
    source = None
    warning = None
    if evaluation_document is None:
        evaluation_document, source, warning = load_evaluation_document(db, resolved_model, storage_client=storage_client)
    else:
        source = "inline"
    if evaluation_document is None:
        evaluation_document = {"model_version": resolved_model}
    evaluation_document = _merge_error_samples(evaluation_document, db, resolved_model, storage_client)
    evaluation_document = _load_class_map(db, resolved_model, evaluation_document, storage_client)
    confusion = build_confusion_report(evaluation_document, model_version=resolved_model)
    manifest_rows = _registry_manifest_rows(db)
    gaps = analyze_data_gaps(manifest_rows, target_config)
    task = generate_collection_task(confusion, gaps, model_version=resolved_model)
    tasks = [task] if task.get("requirements", {}).get("species") else []
    artifact_report = evaluation_document.get("evaluation_artifact_report") if isinstance(evaluation_document, Mapping) else None
    detector_samples = []
    if isinstance(evaluation_document, Mapping):
        raw_detector_samples = evaluation_document.get("detector_samples") or evaluation_document.get("inference_records") or []
        if isinstance(raw_detector_samples, list):
            detector_samples = [row for row in raw_detector_samples if isinstance(row, Mapping)]
    detector_report = analyze_detector_errors(detector_samples)
    return {
        "model": {
            "model_version": resolved_model,
            "evaluation_source": source,
            "evaluation_status": "READY" if source else "MISSING",
            "evaluation_warning": warning,
        },
        "confusion_report": confusion,
        "metrics": _metrics_for_dashboard(evaluation_document),
        "data_gaps": gaps,
        "production_tasks": tasks,
        "manifest": {"source": "registry", "row_count": len(manifest_rows)},
        "evaluation_artifacts": artifact_report or {
            "source": source,
            "available": bool(source),
        },
        "detector_errors": detector_report,
    }


def _artifact_output(root: str | Path, model_version: str) -> Path:
    return Path(root) / model_version


def analyze_and_write_artifacts(
    db: Session,
    *,
    model_version: str | None = None,
    evaluation_path: str | None = None,
    target_config_path: str | None = None,
    output_root: str | Path = "var/intelligence",
    mine_cases: bool = False,
    strict_hard_cases: bool = False,
) -> dict[str, Any]:
    resolved_model = _latest_model_version(db, model_version)
    if evaluation_path:
        source_path = Path(evaluation_path)
        if source_path.is_dir():
            evaluation_path = str(source_path / "metrics.json")
        document = _read_uri_or_path(evaluation_path)
        source = evaluation_path
        document = _merge_evaluation_artifacts(document, source)
    else:
        document, source, warning = load_evaluation_document(db, resolved_model)
        if document is None:
            raise ValueError(warning or "evaluation artifact not found")
    document = _merge_error_samples(document, db, resolved_model)
    document = _load_class_map(db, resolved_model, document)
    confusion = build_confusion_report(document, model_version=resolved_model)
    gaps = analyze_data_gaps(_registry_manifest_rows(db), target_config_path)
    task = generate_collection_task(confusion, gaps, model_version=resolved_model)
    destination = _artifact_output(output_root, resolved_model)
    destination.mkdir(parents=True, exist_ok=True)
    confusion_path = destination / "confusion_report.json"
    gaps_path = destination / "data_gap_report.json"
    task_path = destination / "DATA_PRODUCTION_TASK.json"
    confusion_path.write_text(json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gaps_path.write_text(json.dumps(gaps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_collection_task(task, task_path)
    result: dict[str, Any] = {
        "model_version": resolved_model,
        "source": source,
        "confusion_report": confusion,
        "data_gaps": gaps,
        "task": task,
        "paths": {
            "confusion_report": str(confusion_path),
            "data_gap_report": str(gaps_path),
            "task": str(task_path),
        },
    }
    if mine_cases:
        samples = None
        if isinstance(document, Mapping):
            for key in ("samples", "evaluation_samples", "predictions", "results", "items", "errors"):
                if isinstance(document.get(key), list):
                    samples = document[key]
                    break
        result["hard_cases"] = mine_hard_cases(
            samples or [],
            confusion,
            destination / "hard_cases",
            resolved_model,
            strict=strict_hard_cases,
        )
    return result


@router.get("/intelligence", response_class=HTMLResponse)
def intelligence_page(request: Request):
    return templates.TemplateResponse(request=request, name="intelligence.html", context={})


@router.get("/api/intelligence")
def intelligence_dashboard(
    model_version: str | None = Query(default=None, max_length=128),
    db: Session = Depends(get_db),
):
    try:
        return build_intelligence_payload(db, model_version=model_version)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"模型智能分析失败：{exc}") from exc


@router.get("/api/intelligence/confusion")
def intelligence_confusion(model_version: str | None = Query(default=None, max_length=128), db: Session = Depends(get_db)):
    return intelligence_dashboard(model_version=model_version, db=db)["confusion_report"]


@router.get("/api/intelligence/gaps")
def intelligence_gaps(model_version: str | None = Query(default=None, max_length=128), db: Session = Depends(get_db)):
    return intelligence_dashboard(model_version=model_version, db=db)["data_gaps"]


@router.get("/api/intelligence/tasks")
def intelligence_tasks(model_version: str | None = Query(default=None, max_length=128), db: Session = Depends(get_db)):
    return intelligence_dashboard(model_version=model_version, db=db)["production_tasks"]


@router.post("/api/intelligence/tasks")
def create_intelligence_task(payload: IntelligenceTaskRequest, db: Session = Depends(get_db)):
    dashboard = intelligence_dashboard(model_version=payload.model_version, db=db)
    task = generate_collection_task(
        dashboard["confusion_report"],
        dashboard["data_gaps"],
        task_id=payload.task_id,
        model_version=dashboard["model"]["model_version"],
        scenes=payload.scenes,
        sequence=payload.sequence,
    )
    return {"task": task, "creates_batch": False}


@router.post("/api/intelligence/tasks/{task_id}/batch")
def propose_intelligence_batch(
    task_id: str,
    model_version: str | None = Query(default=None, max_length=128),
    db: Session = Depends(get_db),
):
    dashboard = intelligence_dashboard(model_version=model_version, db=db)
    task = generate_collection_task(
        dashboard["confusion_report"],
        dashboard["data_gaps"],
        task_id=task_id,
        model_version=dashboard["model"]["model_version"],
    )
    return {
        "status": "PROPOSAL_ONLY",
        "message": "采集任务已生成；请在审核后通过现有 Batch Upload 创建批次。",
        "task": task,
        "batch": task["batch_suggestion"],
        "creates_batch": False,
    }


@router.post("/api/intelligence/analyze")
def analyze_intelligence(payload: IntelligenceAnalyzeRequest, db: Session = Depends(get_db)):
    try:
        return analyze_and_write_artifacts(
            db,
            model_version=payload.model_version,
            evaluation_path=payload.evaluation_path,
            target_config_path=payload.target_config_path,
            output_root=payload.output_root,
            mine_cases=payload.mine_hard_cases,
            strict_hard_cases=payload.strict_hard_cases,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"生成 Model Intelligence artifacts 失败：{exc}") from exc


__all__ = [
    "IntelligenceAnalyzeRequest",
    "IntelligenceTaskRequest",
    "analyze_and_write_artifacts",
    "build_intelligence_payload",
    "load_evaluation_document",
    "router",
    "templates",
]
