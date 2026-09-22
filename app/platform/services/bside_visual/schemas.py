from __future__ import annotations

from pydantic import BaseModel, Field


NOT_STARTED = "NOT_STARTED"
STALE = "STALE"

STEP_TRANSPARENT = "transparent"
STEP_STANDARDIZE = "standardize"
STEP_OUTLINE = "outline"
STEP_COMPOSE = "compose"

# The persisted API accepts both the new four-step vocabulary and legacy rows.
# SUCCESS/PROCESSING/ERROR are the public state names; aliases keep older callers
# and historical rows readable during the additive rollout.
PROCESSING = "PROCESSING"
SUCCESS = "SUCCESS"
ERROR = "ERROR"
RUNNING = PROCESSING
COMPLETE = SUCCESS
FAILED = ERROR

STEP_ORDER = (STEP_TRANSPARENT, STEP_STANDARDIZE, STEP_OUTLINE, STEP_COMPOSE)


class StandardizeRequest(BaseModel):
    """Step 2 is automatic; the request is intentionally parameter-free."""

    pass


class OutlineRequest(BaseModel):
    style_id: str = Field(default="lake_mist", min_length=1, max_length=64)


class ComposeRequest(BaseModel):
    template_id: str = Field(default="lake_dawn_01", min_length=1, max_length=64)
