"""Composable B-side processing facade.

The functions in this module operate only on already-created transparent fish
PNG bytes. They never call a detector, segmentation model, GPU worker, or image
generation model.
"""

from .outline_renderer import outline
from .standardizer import standardize, validate_transparent_fish
from .water_renderer import compose_bside

__all__ = ["compose_bside", "outline", "standardize", "validate_transparent_fish"]
