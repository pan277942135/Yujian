"""Small operational view for user-triggered B-side generation jobs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import FishBsideJob
from app.platform.services.bside_assets import read_bside_uri


router = APIRouter(prefix="/api/platform/bside-jobs", tags=["bside-jobs"])


def _asset_url(job_id: str, kind: str) -> str:
    return f"/api/platform/bside-jobs/{job_id}/media/{kind}"


def _job_dict(row: FishBsideJob) -> dict:
    return {
        "id": row.id,
        "fish_record_id": row.fish_record_id,
        "user_id": row.user_id,
        "status": row.status,
        "background_id": row.background_id,
        "outline_style_id": row.outline_style_id,
        "error_code": row.error_code,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        "assets": {
            kind: _asset_url(row.id, kind) if uri else None
            for kind, uri in {
                "input": row.input_image_uri,
                "transparent": row.transparent_fish_uri,
                "standardized": row.standardized_fish_uri,
                "outlined": row.outlined_fish_uri,
                "result": row.result_uri,
            }.items()
        },
    }


@router.get("")
def list_bside_jobs(limit: int = 100, db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(
        select(FishBsideJob).order_by(desc(FishBsideJob.created_at)).limit(max(1, min(int(limit), 200)))
    ).all()
    return {"items": [_job_dict(row) for row in rows]}


@router.get("/{job_id}/media/{kind}")
def job_asset(job_id: str, kind: str, db: Session = Depends(get_db)):
    """Read one already-persisted stage asset for the operational console."""

    row = db.get(FishBsideJob, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="B 面任务不存在")
    uri = {
        "input": row.input_image_uri,
        "transparent": row.transparent_fish_uri,
        "standardized": row.standardized_fish_uri,
        "outlined": row.outlined_fish_uri,
        "result": row.result_uri,
    }.get(kind)
    if not uri:
        raise HTTPException(status_code=404, detail="该阶段尚无资产")
    try:
        content, media_type = read_bside_uri(str(uri))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="任务资产暂时无法读取") from exc
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "private, no-store"})


__all__ = ["router"]
