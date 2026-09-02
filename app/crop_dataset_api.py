from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import get_bucket_name
from trainer.crop_dataset_pipeline import (
    CROP_DATASET_VERSION,
    build_reviewed_crop_dataset_from_db,
    freeze_crop_dataset,
)
from trainer.crop_dataset_validator import CropDatasetValidationError, validate_crop_dataset


router = APIRouter(prefix="/api/crop-datasets", tags=["crop-datasets"])


class CropDatasetBuildRequest(BaseModel):
    dataset_version: str = Field(default=CROP_DATASET_VERSION, min_length=4, max_length=128)
    expand_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    limit: int = Field(default=5000, ge=1, le=100000)
    freeze: bool = False
    git_commit: str | None = Field(default=None, max_length=128)


class CropDatasetFreezeRequest(BaseModel):
    confirm: bool = False
    git_commit: str | None = Field(default=None, max_length=128)


def _safe_version(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"DS_[A-Za-z0-9_.-]{1,120}", value):
        raise HTTPException(status_code=400, detail="dataset_version 格式不合法")
    return value


def _staging_root(dataset_version: str) -> Path:
    base = Path(os.getenv("CROP_DATASET_STAGING_ROOT", "var/crop_datasets"))
    return base / _safe_version(dataset_version)


@router.post("/build")
def build_crop_dataset_endpoint(payload: CropDatasetBuildRequest, db: Session = Depends(get_db)) -> dict:
    """Build reviewed crops; optionally publish/register them when explicitly frozen."""

    version = _safe_version(payload.dataset_version)
    root = _staging_root(version)
    if (root / "metadata" / "crop_manifest.csv").exists():
        raise HTTPException(status_code=409, detail="该 Crop Dataset 已有未覆盖的 staging manifest")
    try:
        report = build_reviewed_crop_dataset_from_db(
            db,
            root,
            dataset_version=version,
            expand_ratio=payload.expand_ratio,
            limit=payload.limit,
        )
        if not report.get("validation", {}).get("valid"):
            raise CropDatasetValidationError(report.get("validation") or {"valid": False})
        result: dict = {
            "dataset_version": version,
            "status": "BUILT_NOT_FROZEN",
            "staging_root": str(root),
            "image_count": report.get("written", 0),
            "class_count": report.get("class_count", 0),
            "validation": report.get("validation"),
            "safety": report.get("safety"),
            "freeze_required": True,
        }
        if payload.freeze:
            result["freeze"] = freeze_crop_dataset(
                root,
                dataset_version=version,
                bucket_name=get_bucket_name(),
                db=db,
                git_commit=(payload.git_commit or os.getenv("APP_GIT_COMMIT", "unknown")).strip() or "unknown",
            )
            result["status"] = "READY_FOR_TRAINING"
        return result
    except CropDatasetValidationError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_INVALID", "report": exc.report}) from exc
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_BUILD_FAILED", "reason": str(exc)}) from exc


@router.get("/{dataset_version}/validation")
def validate_crop_dataset_endpoint(dataset_version: str) -> dict:
    version = _safe_version(dataset_version)
    root = _staging_root(version)
    return {"dataset_version": version, "validation": validate_crop_dataset(root, require_metadata=True, check_source_image=True)}


@router.post("/{dataset_version}/freeze")
def freeze_crop_dataset_endpoint(
    dataset_version: str,
    payload: CropDatasetFreezeRequest,
    db: Session = Depends(get_db),
) -> dict:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Freeze 需要 confirm=true")
    version = _safe_version(dataset_version)
    root = _staging_root(version)
    try:
        return freeze_crop_dataset(
            root,
            dataset_version=version,
            bucket_name=get_bucket_name(),
            db=db,
            git_commit=(payload.git_commit or os.getenv("APP_GIT_COMMIT", "unknown")).strip() or "unknown",
        )
    except CropDatasetValidationError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_INVALID", "report": exc.report}) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"error": "CROP_DATASET_FREEZE_FAILED", "reason": str(exc)}) from exc


__all__ = ["router"]
