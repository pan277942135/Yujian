"""Build an alpha-preserving fish cutout without writing production data."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def build_png_cutout(image: Image.Image, mask: np.ndarray) -> bytes:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)
    output = BytesIO()
    rgba.save(output, format="PNG", optimize=True)
    return output.getvalue()