"""Deterministic CPU-only services for the Qwen B-side visual demo."""

from .outline_renderer import outline
from .standardizer import standardize, validate_transparent_fish
from .water_renderer import compose_bside

__all__ = ["compose_bside", "outline", "standardize", "validate_transparent_fish"]
