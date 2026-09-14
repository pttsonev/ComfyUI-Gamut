"""Tagged colour EXRs containing explicitly named, untransformed data layers.

This writer shares the colour header with write_exr, never with write_data_exr.
Normal XYZ and alpha are data even though the containing file has colour tags.
"""

import json
from os import fspath

import numpy as np
import OpenEXR

from . import alpha, exr


def _channels(passes, halves):
    channels = {}
    shape = None
    for name, arr in passes.items():
        if name not in ("rgb", "albedo", "irradiance", "normal"):
            raise ValueError(f"Unknown EXR pass: {name!r}.")
        pixels = np.asarray(arr)
        allowed = (3, 4) if name == "rgb" else (3,)
        if pixels.ndim != 3 or pixels.shape[-1] not in allowed or 0 in pixels.shape:
            raise ValueError(f"{name} requires nonempty [H,W,{allowed}]; got {pixels.shape}.")
        if not np.issubdtype(pixels.dtype, np.floating):
            raise TypeError(f"{name} requires floating-point pixels.")
        if shape is not None and pixels.shape[:2] != shape:
            raise ValueError(f"{name} expected shape {shape}; got {pixels.shape[:2]}.")
        shape = pixels.shape[:2]
        half = halves[name]
        if not isinstance(half, (bool, np.bool_)):
            raise TypeError(f"{name}_half requires a boolean.")
        components = "XYZ" if name == "normal" else "RGBA"[:pixels.shape[-1]]
        for index, component in enumerate(components):
            channel = component if name == "rgb" else f"{name}.{component}"
            plane = pixels[..., index]
            if half:
                exr._guard_half_range(plane, f"Channel {channel!r}")
            channels[channel] = np.ascontiguousarray(plane, dtype=np.float16 if half else np.float32)
    if "rgb" not in passes:
        raise ValueError("A multilayer colour EXR requires the rgb pass.")
    return channels


def write_multilayer_exr(path, passes, colorspace, *, rgb_half=True, albedo_half=True,
                         irradiance_half=True, normal_half=False, compression="zip"):
    """Write one frame. Colour declaration describes pixels; nothing is converted."""
    header = exr.colour_header(colorspace, compression)
    channels = _channels(passes, dict(rgb=rgb_half, albedo=albedo_half,
                                     irradiance=irradiance_half, normal=normal_half))
    header["gamut:layerRoles"] = json.dumps({
        "schema": 1,
        "colour": ["rgba" if name == "rgb" else name
                   for name in ("rgb", "albedo", "irradiance") if name in passes],
        "data": ["normal"] if "normal" in passes else [],
    }, separators=(",", ":"))
    with OpenEXR.File(header, channels) as image:
        image.write(fspath(path))


def prepare_batch(rgb, colorspace, *, albedo=None, irradiance=None, normal=None,
                  rgb_colorspace=None, albedo_colorspace=None, irradiance_colorspace=None,
                  alpha_mask=None, alpha_image=None, alpha_channel="R", rgb_half=True,
                  albedo_half=True, irradiance_half=True, normal_half=False, compression="zip"):
    """Validate the entire shot before IO, retaining tensors rather than a packed shot copy."""
    exr.colour_header(colorspace, compression)
    alpha.validate_image(rgb, "rgb")
    passes = {"rgb": rgb}
    for name, image in (("albedo", albedo), ("irradiance", irradiance), ("normal", normal)):
        if image is None:
            continue
        alpha.validate_image(image, name, (3,))
        if image.shape[:3] != rgb.shape[:3]:
            raise ValueError(f"{name} expected {tuple(rgb.shape[:3])}; got {tuple(image.shape[:3])}.")
        passes[name] = image
    for name, cs in (("rgb", rgb_colorspace), ("albedo", albedo_colorspace),
                     ("irradiance", irradiance_colorspace)):
        if cs is None:
            continue
        if name not in passes:
            raise ValueError(f"{name}_colorspace was supplied without its image.")
        exr.colour_header(cs, compression)
        if (cs.primaries, cs.transfer) != (colorspace.primaries, colorspace.transfer):
            raise ValueError(f"{name}_colorspace must match the shared colour declaration.")
    opacity = alpha.extract_alpha(rgb, alpha_mask, alpha_image, alpha_channel)
    halves = dict(rgb=rgb_half, albedo=albedo_half, irradiance=irradiance_half, normal=normal_half)
    # Check every frame, including half overflow, before paths/directories are touched.
    # The temporary planes are at most one frame, and are released on each iteration.
    for frame in batch_frames(passes, opacity):
        _channels(frame, halves)
    return passes, opacity


def batch_frames(passes, opacity=None):
    for index in range(passes["rgb"].shape[0]):
        frame = {}
        for name, images in passes.items():
            pixels = images[index].detach().cpu()
            if name == "rgb" and opacity is not None:
                pixels = alpha.embed_alpha(pixels, opacity[index].detach().cpu())
            frame[name] = exr.as_numpy(pixels)
        yield frame
