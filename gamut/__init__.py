"""Gamut's colour declarations and optional-dependency-free colour maths."""

from .colorspace import PRIMARIES, TRANSFERS, ColorSpace, describe

__all__ = ["ColorSpace", "PRIMARIES", "TRANSFERS", "describe"]
