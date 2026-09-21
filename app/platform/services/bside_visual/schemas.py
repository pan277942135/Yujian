from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
STALE = "STALE"

STEP_STANDARDIZE = "standardize"
STEP_OUTLINE = "outline"
STEP_COMPOSE = "compose"

STEP_ORDER = (STEP_STANDARDIZE, STEP_OUTLINE, STEP_COMPOSE)


class StandardizeRequest(BaseModel):
    manual_rotation_offset_deg: float = Field(default=0.0, ge=-15.0, le=15.0)

    @field_validator("manual_rotation_offset_deg")
    @classmethod
    def half_degree_steps(cls, value: float) -> float:
        if abs(value * 2 - round(value * 2)) > 1e-6:
            raise ValueError("manual_rotation_offset_deg 必须以 0.5 度为步进")
        return round(float(value) * 2) / 2


class OutlineRequest(BaseModel):
    style_id: str = Field(default="lake_mist", min_length=1, max_length=64)


class ComposeRequest(BaseModel):
    template_id: str = Field(default="lake_dawn_01", min_length=1, max_length=64)
