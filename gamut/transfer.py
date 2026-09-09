"""Elementwise transfer curves between linear light and ordinary encoded codes.

No curve knows about primaries or a VAE domain. Values are never upper-clamped.
The inactive nonlinear branches are guarded so negative inputs remain safe.
"""

from collections.abc import Callable
from functools import partial

import torch


# ARRI LogC3 EI800, matching the Lightricks HDR implementation.
LOGC3_A = 5.555556
LOGC3_B = 0.052272
LOGC3_C = 0.247190
LOGC3_D = 0.385537
LOGC3_E = 5.367655
LOGC3_F = 0.092809
LOGC3_CUT = 0.010591

# AMPAS S-2016-001 ACEScct.
ACESCCT_A = 10.5402377416545
ACESCCT_B = 0.0729055341958355
ACESCCT_X_BRK = 0.0078125
ACESCCT_Y_BRK = 0.155251141552511
ACESCCT_LOG_M = 17.52
ACESCCT_LOG_B = 9.72


def srgb_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode linear light as sRGB, extending the linear toe below zero."""
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
    )


def srgb_decode(x: torch.Tensor) -> torch.Tensor:
    """Decode sRGB codes without clipping negative values or highlights."""
    return torch.where(
        x <= 0.04044823627710817,
        x / 12.92,
        ((x.clamp_min(0.04044823627710817) + 0.055) / 1.055).pow(2.4),
    )


def logc3_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode linear light as ordinary, unbounded LogC3 EI800 codes."""
    return torch.where(
        x > LOGC3_CUT,
        LOGC3_C * torch.log10(LOGC3_A * x.clamp_min(LOGC3_CUT) + LOGC3_B)
        + LOGC3_D,
        LOGC3_E * x + LOGC3_F,
    )


def logc3_decode(x: torch.Tensor) -> torch.Tensor:
    """Decode ordinary LogC3 codes; no [0, 1] clamp or VAE [-1, 1] rescale."""
    return torch.where(
        x > LOGC3_E * LOGC3_CUT + LOGC3_F,
        (torch.pow(10.0, (x - LOGC3_D) / LOGC3_C) - LOGC3_B) / LOGC3_A,
        (x - LOGC3_F) / LOGC3_E,
    )


def acescct_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode linear light with the ACEScct logarithmic curve and linear toe."""
    return torch.where(
        x <= ACESCCT_X_BRK,
        ACESCCT_A * x + ACESCCT_B,
        (torch.log2(x.clamp_min(ACESCCT_X_BRK)) + ACESCCT_LOG_B) / ACESCCT_LOG_M,
    )


def acescct_decode(x: torch.Tensor) -> torch.Tensor:
    """Decode ACEScct, retaining values outside the nominal code range."""
    return torch.where(
        x <= ACESCCT_Y_BRK,
        (x - ACESCCT_B) / ACESCCT_A,
        torch.pow(2.0, x * ACESCCT_LOG_M - ACESCCT_LOG_B),
    )


def gamma_encode(x: torch.Tensor, g: float) -> torch.Tensor:
    """Encode with reciprocal gamma, preserving the sign of negative light."""
    return torch.sign(x) * torch.abs(x).pow(1.0 / g)


def gamma_decode(x: torch.Tensor, g: float) -> torch.Tensor:
    """Decode with gamma, preserving the sign of negative codes."""
    return torch.sign(x) * torch.abs(x).pow(g)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


_ENCODERS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": _identity,
    "srgb": srgb_encode,
    "logc3": logc3_encode,
    "acescct": acescct_encode,
    "g22": partial(gamma_encode, g=2.2),
    "g24": partial(gamma_encode, g=2.4),
}

_DECODERS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "linear": _identity,
    "srgb": srgb_decode,
    "logc3": logc3_decode,
    "acescct": acescct_decode,
    "g22": partial(gamma_decode, g=2.2),
    "g24": partial(gamma_decode, g=2.4),
}


def encode(
    img: torch.Tensor, curve: str, clamp_negatives: bool = False
) -> torch.Tensor:
    """Encode by registry name; optionally clamp negative linear input to zero."""
    try:
        transform = _ENCODERS[curve]
    except KeyError:
        raise KeyError(f"Unknown transfer curve: {curve!r}") from None
    return transform(img.clamp_min(0.0) if clamp_negatives else img)


def decode(
    img: torch.Tensor, curve: str, clamp_negatives: bool = False
) -> torch.Tensor:
    """Decode by registry name; optionally clamp negative linear output to zero."""
    try:
        transform = _DECODERS[curve]
    except KeyError:
        raise KeyError(f"Unknown transfer curve: {curve!r}") from None
    linear = transform(img)
    return linear.clamp_min(0.0) if clamp_negatives else linear


def transform_rgb(img: torch.Tensor, curve: str, direction: str) -> torch.Tensor:
    """Apply a transfer to RGB only, passing alpha and any extra channels through."""
    if img.ndim == 0 or img.shape[-1] < 3:
        raise ValueError("Transfer conversion requires [..., C] with C >= 3.")
    if direction not in ("encode", "decode"):
        raise ValueError(f"Unknown transfer direction: {direction!r}")
    transform = encode if direction == "encode" else decode
    rgb = transform(img[..., :3], curve)
    return rgb if img.shape[-1] == 3 else torch.cat((rgb, img[..., 3:]), dim=-1)
