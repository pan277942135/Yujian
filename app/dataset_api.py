"""Dataset Freeze API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import DatasetVersion, ImageAsset
from app.dataset_freeze_service import FreezeConfig, build_manifest, serialize_metadata

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetFreezeRequest(BaseModel):
    dataset_version: str
    parent_version: str | None = None


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    rows = db.scalars(select(DatasetVersion).order_by(DatasetVersion.created_at.desc())).all()
    return [
        {
            "dataset_version": row.dataset_version,
            "status": row.status,
            "train_count": row.train_count,
            "val_count": row.val_count,
            "test_count": row.test_count,
            "species_count": row.species_count,
            "manifest_uri": row.manifest_uri,
        }
        for row in rows
    ]


@router.post("/freeze/preview")
def preview_freeze(payload: DatasetFreezeRequest, db: Session = Depends(get_db)):
    config = FreezeConfig(dataset_version=payload.dataset_version)
    images = db.scalars(select(ImageAsset)).all()
    metadata = serialize_metadata(build_manifest(images, config), config)
    return {
        "dataset_version": payload.dataset_version,
        "status": "PREVIEW",
        "image_count": metadata["report"]["total_images"],
        "species_count": metadata["report"]["species_count"],
        "report": metadata["report"],
    }


@router.post("/freeze")
def freeze_dataset(payload: DatasetFreezeRequest, db: Session = Depends(get_db)):
    exists = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_version == payload.dataset_version))
    if exists:
        raise HTTPException(status_code=409, detail="dataset version already exists")

    config = FreezeConfig(dataset_version=payload.dataset_version)
    images = db.scalars(select(ImageAsset)).all()
    metadata = serialize_metadata(build_manifest(images, config), config)

    report = metadata["report"]
    row = DatasetVersion(
        dataset_version=payload.dataset_version,
        parent_version=payload.parent_version,
        manifest_uri=f"dataset://{payload.dataset_version}/manifest.json",
        class_map_uri=f"dataset://{payload.dataset_version}/class_mapping.json",
        train_count=report["split_distribution"].get("train", 0),
        val_count=report["split_distribution"].get("val", 0),
        test_count=report["split_distribution"].get("test", 0),
        species_count=report["species_count"],
        git_commit="pending",
        selection_mode="ALL_APPROVED_SINGLE_FISH",
        source_cutoff_at=datetime.now(timezone.utc),
        status="FROZEN",
    )
    db.add(row)
    db.commit()

    return {
        "dataset_version": payload.dataset_version,
        "status": "FROZEN",
        "report": report,
    }
