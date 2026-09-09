"""Optional OpenColorIO CPU bridge; importing this module never imports OCIO."""

import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"


@lru_cache(maxsize=1)
def available() -> bool:
    """Probe the optional binding once, including failure to load its library."""
    try:
        importlib.import_module("PyOpenColorIO")
    except (ImportError, OSError):
        return False
    return True


def _ocio() -> Any:
    if not available():
        raise ImportError(
            "OpenColorIO is unavailable; install the optional opencolorio package."
        )
    return importlib.import_module("PyOpenColorIO")


def get_config(spec: str | os.PathLike[str] = DEFAULT_CONFIG) -> Any:
    """Load a built-in config name, an explicit file, or the $OCIO file path."""
    ocio = _ocio()
    spec = os.fspath(spec)
    if spec == "$OCIO":
        path = os.environ.get("OCIO")
        if not path:
            raise ValueError(
                "$OCIO requested but the OCIO environment variable is unset."
            )
        return ocio.Config.CreateFromFile(os.path.expanduser(os.path.expandvars(path)))
    expanded = os.path.expanduser(os.path.expandvars(spec))
    if spec.startswith("ocio://"):
        return ocio.Config.CreateFromBuiltinConfig(spec[len("ocio://"):])
    if Path(expanded).is_file() or Path(expanded).suffix == ".ocio" or any(
        separator in expanded for separator in ("/", "\\")
    ):
        return ocio.Config.CreateFromFile(expanded)
    return ocio.Config.CreateFromBuiltinConfig(spec)


def list_spaces(cfg: Any) -> list[str]:
    """Return the config's selectable colour space names."""
    _ocio()
    return list(cfg.getColorSpaceNames())


def _apply(arr: np.ndarray, processor: Any) -> np.ndarray:
    ocio = _ocio()
    pixels = np.array(arr, dtype=np.float32, order="C", copy=True)
    if pixels.ndim == 0 or pixels.shape[-1] not in (3, 4):
        raise ValueError("OCIO requires [..., 3 or 4] RGB/RGBA pixels.")
    if pixels.size:
        channels = pixels.shape[-1]
        width = pixels.shape[-2] if pixels.ndim > 1 else 1
        height = pixels.size // (width * channels)
        description = ocio.PackedImageDesc(pixels, width, height, channels)
        processor.apply(description)
    return pixels


def apply_transform(arr: np.ndarray, cfg: Any, src: str, dst: str) -> np.ndarray:
    """Apply a colour-space transform to a contiguous float32 copy."""
    _ocio()
    processor = cfg.getProcessor(src, dst).getDefaultCPUProcessor()
    return _apply(arr, processor)


def apply_display(
    arr: np.ndarray, cfg: Any, src: str, display: str, view: str
) -> np.ndarray:
    """Apply a display/view with an explicit source; never guess scene_linear."""
    if not isinstance(src, str) or not src.strip():
        raise ValueError("apply_display requires an explicit source colour space.")
    ocio = _ocio()
    transform = ocio.DisplayViewTransform(src=src, display=display, view=view)
    processor = cfg.getProcessor(transform).getDefaultCPUProcessor()
    return _apply(arr, processor)


def apply_file_transform(
    arr: np.ndarray, path: str | os.PathLike[str], direction: str = "forward"
) -> np.ndarray:
    """Apply a LUT/file transform in the explicit forward or inverse direction."""
    ocio = _ocio()
    directions = {
        "forward": ocio.TRANSFORM_DIR_FORWARD,
        "inverse": ocio.TRANSFORM_DIR_INVERSE,
    }
    if direction not in directions:
        raise ValueError(f"Unknown file transform direction: {direction!r}")
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(os.fspath(path))))
    transform = ocio.FileTransform(src=path, direction=directions[direction])
    processor = ocio.Config.CreateRaw().getProcessor(transform).getDefaultCPUProcessor()
    return _apply(arr, processor)
