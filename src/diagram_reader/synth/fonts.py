"""Chargement de polices TrueType avec repli sur la police par défaut de Pillow."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from PIL import ImageFont

_CANDIDATES_REGULAR = (
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "FreeSans.ttf",
    "Arial.ttf",
    "arial.ttf",
)
_CANDIDATES_BOLD = (
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "FreeSansBold.ttf",
    "Arial Bold.ttf",
    "arialbd.ttf",
)
_SEARCH_DIRS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("C:/Windows/Fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
)


@cache
def _find(name: str) -> str | None:
    for base in _SEARCH_DIRS:
        if not base.exists():
            continue
        hits = list(base.rglob(name))
        if hits:
            return str(hits[0])
    return None


@cache
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Retourne une police de la taille demandée (en pixels)."""
    size = max(6, int(size))
    for name in _CANDIDATES_BOLD if bold else _CANDIDATES_REGULAR:
        path = _find(name)
        if path:
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()
