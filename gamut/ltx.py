"""Adapter for the LTX HDR IC-LoRA LogC3 VAE output contract."""

import torch

from . import primaries, transfer
from .colorspace import ColorSpace


def decode_ltx_hdr(
    image: torch.Tensor,
    target_primaries: str = "rec709",
    exposure: float = 0.0,
    clamp_negatives: bool = True,
) -> tuple[torch.Tensor, ColorSpace]:
    """Decode VAE [0, 1] codes, apply exposure in stops, and rotate primaries.

    Only this adapter clamps the VAE input to [0, 1]. There is deliberately no
    *2-1 rescale: the upstream wrapper's rescale and LogC3.decompress's inverse
    rescale cancel. The generic LogC3 curve therefore receives ordinary codes.
    Output highlights have no upper clamp; optional negative clipping is last.
    """
    native = ColorSpace("rec709", "linear", "node:GamutLTXHDRDecode")
    linear = transfer.logc3_decode(image.clamp(0.0, 1.0))
    linear = linear * (2.0**exposure)
    linear = primaries.convert(linear, "rec709", target_primaries, colorspace=native)
    if clamp_negatives:
        linear = linear.clamp_min(0.0)
    return linear, ColorSpace(target_primaries, "linear", native.source)
