"""Tagged RGB/RGBA EXR IO using the OpenEXR 3.x File API."""

from os import PathLike, fspath

import numpy as np
import OpenEXR

from .colorspace import TRANSFERS, ColorSpace
from .primaries import CHROMATICITIES


_COMPRESSIONS = {
    "none": OpenEXR.NO_COMPRESSION,
    "rle": OpenEXR.RLE_COMPRESSION,
    "zips": OpenEXR.ZIPS_COMPRESSION,
    "zip": OpenEXR.ZIP_COMPRESSION,
    "piz": OpenEXR.PIZ_COMPRESSION,
}

# Complete space names used by other writers. A transfer-only tag such as
# "Linear" cannot determine primaries without chromaticities.
_NAMED_SPACES = {
    "acescg": ("acescg", "linear"),
    "aces2065-1": ("ap0", "linear"),
    "aces2065-1 (ap0)": ("ap0", "linear"),
    "linear rec.709 (srgb)": ("rec709", "linear"),
    "linear rec.2020": ("rec2020", "linear"),
    "srgb": ("rec709", "srgb"),
    "acescct": ("acescg", "acescct"),
    "logc3": ("rec709", "logc3"),
    "arri logc3 (ei800)": ("rec709", "logc3"),
}

# Written tag for each (primaries, transfer) pair we can name completely.
# A complete name is what Nuke and OCIO key off; the chromaticities carry the
# primaries regardless, so this only ever adds information. Vocabulary matches
# Lightricks' ltx_core (ACEScg / ACEScct / LogC3 / sRGB) and the OCIO ACES
# config's space names, so a written file round-trips through both.
_CANONICAL_NAMES = {
    ("acescg", "linear"): "ACEScg",
    ("ap0", "linear"): "ACES2065-1",
    ("rec709", "linear"): "Linear Rec.709 (sRGB)",
    ("rec2020", "linear"): "Linear Rec.2020",
    ("rec709", "srgb"): "sRGB",
    ("acescg", "acescct"): "ACEScct",
    ("rec709", "logc3"): "LogC3",
}
_TRANSFER_NAMES = {name.casefold(): key for key, name in TRANSFERS.items()}
_TRANSFER_NAMES.update({key: key for key in TRANSFERS})


def _infer_colorspace(header: dict) -> ColorSpace | None:
    tag = header.get("colorSpace")
    name = tag.strip().casefold() if isinstance(tag, str) else ""
    if "chromaticities" in header:
        chroma = np.asarray(header["chromaticities"], dtype=np.float64)
        if chroma.shape != (8,):
            return None
        primaries = next(
            (key for key, xy in CHROMATICITIES.items()
             if np.allclose(chroma, xy, atol=1e-4, rtol=0)),
            None,
        )
        if primaries is None:
            return None
        if tag is None:
            # EXR's normal linear-light convention for chromaticities-only files.
            transfer = "linear"
        elif name in _TRANSFER_NAMES:
            transfer = _TRANSFER_NAMES[name]
        elif name in _NAMED_SPACES:
            named_primaries, transfer = _NAMED_SPACES[name]
            if named_primaries != primaries:
                return None
        else:
            return None
        return ColorSpace(primaries, transfer, "exr:chromaticities")
    if name in _NAMED_SPACES:
        primaries, transfer = _NAMED_SPACES[name]
        return ColorSpace(primaries, transfer, "exr:colorSpace")
    return None


def read_exr(path: str | PathLike[str]) -> tuple[np.ndarray, ColorSpace | None, dict]:
    """Read one flat RGB/RGBA image as float32, its declaration, and raw header.

    Untagged or unrecognised colour metadata returns None, leaving fallback
    choices to the caller. Unknown chromaticities remain in the raw header.
    Multipart, deep, and non-RGB files are rejected rather than misread.
    """
    with OpenEXR.File(fspath(path), separate_channels=True) as image:
        if len(image.parts) != 1:
            raise ValueError("read_exr requires a single-part RGB or RGBA file.")
        header = dict(image.header())
        if header["type"] not in (OpenEXR.scanlineimage, OpenEXR.tiledimage):
            raise ValueError("read_exr does not support deep EXR data.")
        channels = image.channels()
        if not all(name in channels for name in "RGB"):
            raise ValueError("read_exr requires R, G, and B channels.")
        names = "RGBA" if "A" in channels else "RGB"
        pixels = np.stack([channels[name].pixels for name in names], axis=-1)
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    return pixels, _infer_colorspace(header), header


def write_exr(
    path: str | PathLike[str],
    arr: np.ndarray,
    colorspace: ColorSpace,
    half: bool = True,
    compression: str = "zip",
) -> None:
    """Write HWC float RGB/RGBA and real chromaticities/colorSpace attributes.

    Pixels are not clipped or colour-converted. The transfer display name is
    written verbatim. Errors from validation or the EXR writer propagate.
    """
    pixels = np.asarray(arr)
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or 0 in pixels.shape:
        raise ValueError("write_exr requires a nonempty [H, W, 3 or 4] image.")
    if not np.issubdtype(pixels.dtype, np.floating):
        raise TypeError("write_exr requires floating-point pixels.")
    try:
        chromaticities = CHROMATICITIES[colorspace.primaries]
        transfer_name = TRANSFERS[colorspace.transfer]
    except KeyError as exc:
        raise ValueError(f"Unknown EXR colour declaration: {exc.args[0]!r}") from None
    try:
        codec = _COMPRESSIONS[compression.lower()]
    except KeyError:
        raise ValueError(f"Unknown EXR compression: {compression!r}") from None
    pixels = np.ascontiguousarray(pixels, dtype=np.float16 if half else np.float32)
    header = {
        "type": OpenEXR.scanlineimage,
        "compression": codec,
        "chromaticities": tuple(float(value) for value in chromaticities),
        "colorSpace": _CANONICAL_NAMES.get(
            (colorspace.primaries, colorspace.transfer), transfer_name
        ),
    }
    name = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    with OpenEXR.File(header, {name: pixels}) as image:
        image.write(fspath(path))
