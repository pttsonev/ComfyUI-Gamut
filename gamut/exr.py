"""Tagged colour and untagged data-channel EXR IO using the OpenEXR 3.x File API.

Two independent paths live here and must stay independent.

The **colour** path (`read_exr` / `write_exr`) always carries a real colour
declaration: `chromaticities` plus a `colorSpace` name, validated against each
other so a tag can never contradict the primaries beside it.

The **data** path (`read_data_exr` / `write_data_exr`) is for channels that are
not colour — surface normals, depth, coverage, arbitrary AOVs. It accepts no
`ColorSpace` and **never emits `chromaticities` or `colorSpace`**. That absence
is the invariant: it is how a reader tells a data file from a tagged colour
file, and it is why the two writers share only dtype and compression helpers,
never metadata logic. Mixed files (exr_layers.py) are tagged colour containers
with explicitly named data layers: file-level tags describe their RGB layers,
not XYZ or alpha. Absence of tags still identifies standalone data exports;
their presence does not turn every channel in a mixed file into colour.
"""

from collections.abc import Mapping
import json
import struct
from os import PathLike, fspath

import numpy as np
import OpenEXR
import torch

from . import primaries, transfer
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
    # LTX-2 reuses the LogC3 curve on Rec.709 primaries. With chromaticities
    # present this is validated against them, so an ARRI file carrying AWG3
    # chromaticities under the same tag is correctly rejected rather than
    # claimed as Rec.709.
    "logc3": ("rec709", "logc3"),
}

# Transfers that say nothing about primaries. Only these may be read back from
# a bare transfer tag, and only these may be written alongside arbitrary
# chromaticities. Everything else names a complete space whose primaries are
# part of its definition — "ACEScct" means AP1, "sRGB" means Rec.709 — so a
# bare match would let a tag contradict the chromaticities beside it.
#
# Deliberately absent: "ARRI LogC3 (EI800)". ARRI's camera space is the LogC3
# curve on ARRI Wide Gamut 3, NOT Rec.709. LTX-2 reuses the LogC3 *curve* on
# Rec.709 primaries, which is a different colour space that happens to share a
# transfer. We hold no AWG3 matrices, so we refuse to recognise the camera
# space rather than silently misidentify it as Rec.709.
_PRIMARIES_NEUTRAL = frozenset({"linear", "g22", "g24"})

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
        elif name in _NAMED_SPACES:
            # Complete names are checked FIRST and must agree with the
            # chromaticities; a bare-transfer match here would skip that check.
            named_primaries, transfer = _NAMED_SPACES[name]
            if named_primaries != primaries:
                return None
        elif name in _TRANSFER_NAMES and _TRANSFER_NAMES[name] in _PRIMARIES_NEUTRAL:
            transfer = _TRANSFER_NAMES[name]
        else:
            return None
        return ColorSpace(primaries, transfer, "exr:chromaticities")
    if name in _NAMED_SPACES:
        primaries, transfer = _NAMED_SPACES[name]
        return ColorSpace(primaries, transfer, "exr:colorSpace")
    return None


def _guard_half_range(pixels: np.ndarray, what: str) -> None:
    """Refuse a half write that would turn a finite value into inf.

    Shared by both writers so the two paths cannot drift apart on the one
    check that silently destroys data. `what` names the offender for the
    caller — the image on the colour path, the channel on the data path.
    """
    finite = pixels[np.isfinite(pixels)]
    if finite.size and np.abs(finite).max() > 65504.0:
        raise ValueError(
            f"{what} value {float(np.abs(finite).max()):.6g} exceeds the float16 "
            "maximum of 65504; writing half would silently turn finite values "
            "into inf. Set half=False to write float32."
        )


def read_exr(path: str | PathLike[str], layer: str = "") -> tuple[np.ndarray, ColorSpace | None, dict]:
    """Read one flat RGB/RGBA image as float32, its declaration, and raw header.

    Untagged or unrecognised colour metadata returns None, leaving fallback
    choices to the caller. Unknown chromaticities remain in the raw header.
    Multipart, deep, and non-RGB files are rejected rather than misread.
    """
    pixels, header = _read_layer(path, layer)
    return pixels, _infer_colorspace(header), header


def write_exr(
    path: str | PathLike[str],
    arr: np.ndarray,
    colorspace: ColorSpace,
    half: bool = True,
    compression: str = "zip",
) -> None:
    """Write HWC float RGB/RGBA and real chromaticities/colorSpace attributes.

    Pixels are not clipped or colour-converted. Complete canonical space names
    are used where known, with the transfer display name as fallback. Errors
    from validation or the EXR writer propagate.
    """
    pixels = np.asarray(arr)
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or 0 in pixels.shape:
        raise ValueError("write_exr requires a nonempty [H, W, 3 or 4] image.")
    if not np.issubdtype(pixels.dtype, np.floating):
        raise TypeError("write_exr requires floating-point pixels.")
    header = colour_header(colorspace, compression)
    if half:
        _guard_half_range(pixels, "Image")
    pixels = np.ascontiguousarray(pixels, dtype=np.float16 if half else np.float32)
    name = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    with OpenEXR.File(header, {name: pixels}) as image:
        image.write(fspath(path))


def colour_header(colorspace: ColorSpace, compression: str = "zip") -> dict:
    """Validate/build tagged colour metadata. Never used by the data writer."""
    try:
        chromaticities = CHROMATICITIES[colorspace.primaries]
        transfer_name = TRANSFERS[colorspace.transfer]
    except KeyError as exc:
        raise ValueError(f"Unknown EXR colour declaration: {exc.args[0]!r}") from None
    try:
        codec = _COMPRESSIONS[compression.lower()]
    except KeyError:
        raise ValueError(f"Unknown EXR compression: {compression!r}") from None
    pair = (colorspace.primaries, colorspace.transfer)
    if pair not in _CANONICAL_NAMES and colorspace.transfer not in _PRIMARIES_NEUTRAL:
        raise ValueError(
            f"Refusing to write {colorspace.transfer!r} on {colorspace.primaries!r} "
            f"primaries: {TRANSFERS[colorspace.transfer]!r} names a complete colour "
            "space whose own primaries differ, so the colorSpace tag would "
            "contradict the chromaticities. Convert primaries first, or use a "
            "primaries-neutral transfer."
        )

    return {
        "type": OpenEXR.scanlineimage,
        "compression": codec,
        "chromaticities": tuple(float(value) for value in chromaticities),
        "colorSpace": _CANONICAL_NAMES.get(
            (colorspace.primaries, colorspace.transfer), transfer_name
        ),
    }


def read_data_exr(path: str | PathLike[str]) -> tuple[dict[str, np.ndarray], dict]:
    """Read flat named channels as float32 and return the raw header.

    Channel names are preserved, including RGB names. Colour metadata is never
    inferred or interpreted. Multipart and deep files are rejected.
    """
    with OpenEXR.File(fspath(path), separate_channels=True) as image:
        if len(image.parts) != 1:
            raise ValueError("read_data_exr requires a single-part file.")
        header = dict(image.header())
        if header["type"] not in (OpenEXR.scanlineimage, OpenEXR.tiledimage):
            raise ValueError("read_data_exr does not support deep EXR data.")
        channels = {
            name: np.ascontiguousarray(channel.pixels, dtype=np.float32)
            for name, channel in image.channels().items()
        }
    return channels, header


def write_data_exr(
    path: str | PathLike[str],
    channels: Mapping[str, np.ndarray],
    *,
    half: bool = True,
    compression: str = "zip",
) -> None:
    """Write named 2-D float channels without chromaticities or colorSpace.

    All channels must share one nonempty shape. Values are not clipped or
    colour-converted, and no colour declaration is accepted. Errors from
    validation or the EXR writer propagate.
    """
    if not isinstance(channels, Mapping):
        raise TypeError("write_data_exr requires a mapping of channel names to arrays.")
    if not channels:
        raise ValueError("write_data_exr requires at least one channel.")
    if not isinstance(half, (bool, np.bool_)):
        raise TypeError("write_data_exr requires a boolean half option.")
    try:
        codec = _COMPRESSIONS[compression.lower()]
    except KeyError:
        raise ValueError(f"Unknown EXR compression: {compression!r}") from None

    shape = None
    pixels_by_name = {}
    for name, arr in channels.items():
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("write_data_exr requires nonempty channel names without NUL.")
        pixels = np.asarray(arr)
        if pixels.ndim != 2 or 0 in pixels.shape:
            raise ValueError(f"Channel {name!r} requires a nonempty [H, W] array.")
        if not np.issubdtype(pixels.dtype, np.floating):
            raise TypeError(f"Channel {name!r} requires floating-point pixels.")
        if shape is None:
            shape = pixels.shape
        elif pixels.shape != shape:
            raise ValueError("write_data_exr requires all channels to share one shape.")
        if half:
            _guard_half_range(pixels, f"Channel {name!r}")
        pixels_by_name[name] = np.ascontiguousarray(
            pixels, dtype=np.float16 if half else np.float32
        )
    header = {
        "type": OpenEXR.scanlineimage,
        "compression": codec,
    }
    with OpenEXR.File(header, pixels_by_name) as image:
        image.write(fspath(path))


def load_image(
    path: str | PathLike[str],
    fallback_primaries: str = "rec709",
    fallback_transfer: str = "linear",
    tonemap_preview: bool = False,
    colorspace: ColorSpace | None = None,
    layer: str = "",
) -> tuple[torch.Tensor, torch.Tensor, ColorSpace]:
    """Adapt EXR pixels to IMAGE/MASK tensors; real header metadata is authoritative.

    The optional preview decodes, rotates to Rec.709, clips to display range,
    and encodes sRGB. It is a display preview, not an HDR-preserving output.
    ComfyUI's MASK is transparency (one minus alpha).
    """
    pixels, declared, _ = read_exr(path, layer)
    cs = declared or colorspace or ColorSpace(
        fallback_primaries, fallback_transfer, "widget:fallback"
    )
    image = torch.from_numpy(pixels).unsqueeze(0)
    rgb = image[..., :3]
    mask = 1.0 - image[..., 3] if image.shape[-1] == 4 else torch.zeros_like(rgb[..., 0])
    if tonemap_preview:
        linear = transfer.decode(rgb, cs.transfer)
        linear = primaries.convert(linear, cs.primaries, "rec709")
        rgb = transfer.srgb_encode(linear.clamp(0.0, 1.0))
        cs = ColorSpace("rec709", "srgb", "node:GamutLoadEXR:preview")
    return rgb.contiguous(), mask.contiguous(), cs


def as_numpy(pixels):
    """Detach a frame for IO, including torch bfloat16 which NumPy cannot represent."""
    if isinstance(pixels, torch.Tensor):
        pixels = pixels.detach().cpu()
        if pixels.dtype == torch.bfloat16:
            pixels = pixels.float()
        return pixels.numpy()
    return np.asarray(pixels)


def validate_half_batch(image, half=True, what="Image"):
    """Check a shot without constructing a CPU copy of the entire batch."""
    if half:
        for frame in image:
            _guard_half_range(as_numpy(frame), what)


def _flat_header(image):
    if len(image.parts) != 1:
        raise ValueError("EXR reading requires a single-part file.")
    header = dict(image.header())
    if header["type"] not in (OpenEXR.scanlineimage, OpenEXR.tiledimage):
        raise ValueError("EXR reading does not support deep EXR data.")
    return header


def inspect_exr(path):
    """Read only a flat file's header, retaining raw tags and channel descriptors."""
    with OpenEXR.File(fspath(path), separate_channels=True, header_only=True) as image:
        header = _flat_header(image)
    layer_roles(header)
    return header


def channel_types(path):
    """Read chlist pixel types without decoding pixels.

    OpenEXR 3.4.15's header_only Channel objects lose pixel type (their empty
    arrays are float64 and type() raises). Read just the chlist attribute's
    fixed-format records; all image validation remains with OpenEXR.File.
    """
    def cstring(stream):
        value = bytearray()
        for _ in range(256):  # EXR names have at most 255 bytes
            char = stream.read(1)
            if char == b"\0":
                return bytes(value)
            if not char:
                break
            value.extend(char)
        raise ValueError("Invalid EXR header string.")

    with open(path, "rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(8)  # magic + version; inspect_exr validates the file first
        while stream.tell() < size:
            name = cstring(stream)
            if not name:
                break
            kind = cstring(stream)
            raw_size = stream.read(4)
            if len(raw_size) != 4:
                break
            length = struct.unpack("<i", raw_size)[0]
            if length < 0 or stream.tell() + length > size:
                break
            end = stream.tell() + length
            if name == b"channels" and kind == b"chlist":
                result = {}
                while stream.tell() < end:
                    channel = cstring(stream)
                    if not channel:
                        return result
                    record = stream.read(16)  # pixelType, pLinear/reserved, x/ySampling
                    if len(record) != 16 or stream.tell() > end:
                        break
                    pixel_type = struct.unpack("<i", record[:4])[0]
                    if pixel_type not in (0, 1, 2):
                        break
                    result[channel.decode("utf-8")] = ("UINT", "HALF", "FLOAT")[pixel_type]
                break
            stream.seek(end)
    raise ValueError("Invalid EXR channels header.")


def channel_groups(header):
    groups = {}
    for channel in header["channels"]:
        layer, _, component = channel.name.rpartition(".")
        groups.setdefault(layer, {})[component] = channel
    return groups


def layer_roles(header):
    """Validate Gamut's optional role annotation; external files need not have it."""
    if "gamut:layerRoles" not in header:
        return {}
    raw = header["gamut:layerRoles"]
    try:
        roles = json.loads(raw) if isinstance(raw, str) else None
        if not isinstance(roles, dict) or set(roles) != {"schema", "colour", "data"}:
            raise ValueError("expected schema, colour and data fields")
        if type(roles["schema"]) is not int or roles["schema"] != 1:
            raise ValueError("unsupported schema")
        groups = channel_groups(header)
        if "" in groups and "rgba" in groups:
            raise ValueError("ambiguous rgba layer")
        available = {"rgba" if name == "" else name: parts for name, parts in groups.items()}
        result = {}
        for role in ("colour", "data"):
            if not isinstance(roles[role], list):
                raise ValueError(f"{role} must be a list")
            for name in roles[role]:
                if not isinstance(name, str) or not name or name not in available:
                    raise ValueError(f"unknown layer {name!r}")
                if name in result:
                    raise ValueError(f"duplicate or conflicting role for layer {name!r}")
                parts = available[name]
                if role == "colour" and (not set("RGB") <= parts.keys() or set("XYZ") & parts.keys()):
                    raise ValueError(f"colour layer {name!r} requires RGB, not XYZ components")
                result[name] = role
        if set(result) != set(available):
            raise ValueError(f"missing roles for layers {sorted(set(available) - set(result))}")
        return result
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid gamut:layerRoles: {exc}") from None


def _selected_names(header, layer, components=None):
    if not isinstance(layer, str) or "\x00" in layer:
        raise ValueError("Supply an exact layer name without NUL, or empty for root RGB.")
    groups = channel_groups(header)
    roles = layer_roles(header)
    parts = groups.get(layer, {})
    if components is None:
        if roles.get("rgba" if not layer else layer) == "data" or set("XYZ") <= parts.keys():
            raise ValueError(
                f"Layer {layer or 'rgba'!r} is data, not colour R, G, and B; use Gamut: Load Data EXR."
            )
        components = "RGBA" if "A" in parts else "RGB"
    elif components not in ("XYZ", "RGB"):
        raise ValueError("Data components must be XYZ or RGB.")
    if not all(name in parts for name in components):
        required = "R, G, and B" if components.startswith("RGB") else "X, Y, and Z"
        listing = {name or "(root)": sorted(channels) for name, channels in groups.items()}
        raise ValueError(f"Layer {layer!r} requires {required} channels. Available layers: {listing}")
    names = []
    for name in components:
        channel = parts[name]
        if channel.xSampling != 1 or channel.ySampling != 1:
            raise ValueError(f"Subsampled channel {channel.name!r} is not supported.")
        names.append(channel.name)
    return tuple(names)


def _window_signature(header):
    return tuple(tuple(int(value) for corner in header[key] for value in corner)
                 for key in ("dataWindow", "displayWindow"))


def _read_layer(path, layer, components=None):
    header = inspect_exr(path)
    names = _selected_names(header, layer, components)
    with OpenEXR.File(fspath(path), separate_channels=True) as image:
        actual = _flat_header(image)
        if (_selected_names(actual, layer, components) != names
                or _window_signature(actual) != _window_signature(header)):
            raise ValueError(f"EXR changed while reading: {path}")
        pixels = np.stack([image.channels()[name].pixels for name in names], axis=-1)
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    return pixels, actual


def load_sequence(paths, fallback_primaries="rec709", fallback_transfer="linear",
                  tonemap_preview=False, colorspace=None, layer="", components=None):
    """Preflight all frames, then fill a single selected-layer batch. Data returns IMAGE only."""
    if not paths:
        raise ValueError("EXR sequence requires at least one frame.")
    expected = None
    spaces = []
    for path in paths:
        header = inspect_exr(path)
        names = _selected_names(header, layer, components)
        signature = names, _window_signature(header)
        if expected is not None and signature != expected:
            raise ValueError(f"Sequence channels or data/display windows differ at {path}.")
        expected = signature
        if components is None:
            cs = _infer_colorspace(header) or colorspace or ColorSpace(
                fallback_primaries, fallback_transfer, "widget:fallback"
            )
            if spaces and (cs.primaries, cs.transfer) != (spaces[0].primaries, spaces[0].transfer):
                raise ValueError(f"Sequence colour declaration differs at {path}.")
            spaces.append(cs)
    xmin, ymin, xmax, ymax = expected[1][0]
    height, width = ymax - ymin + 1, xmax - xmin + 1
    output = torch.empty((len(paths), height, width, 3), dtype=torch.float32)
    mask = torch.empty((len(paths), height, width), dtype=torch.float32) if components is None else None
    for index, path in enumerate(paths):
        pixels, header = _read_layer(path, layer, components)
        if (_selected_names(header, layer, components), _window_signature(header)) != expected:
            raise ValueError(f"EXR changed after sequence preflight: {path}")
        image = torch.from_numpy(pixels)
        if components is None:
            cs = _infer_colorspace(header) or colorspace or ColorSpace(
                fallback_primaries, fallback_transfer, "widget:fallback"
            )
            if cs != spaces[index]:
                raise ValueError(f"EXR colour declaration changed after sequence preflight: {path}")
            rgb = image[..., :3]
            mask[index] = 1.0 - image[..., 3] if pixels.shape[-1] == 4 else 0.0
            if tonemap_preview:
                linear = primaries.convert(transfer.decode(rgb, cs.transfer), cs.primaries, "rec709")
                rgb = transfer.srgb_encode(linear.clamp(0.0, 1.0))
            output[index] = rgb
        else:
            output[index] = image
    if components is not None:
        return (output,)
    cs = spaces[0]
    if tonemap_preview:
        cs = ColorSpace("rec709", "srgb", "node:GamutLoadEXR:preview")
    elif any(space.source != cs.source for space in spaces):
        cs = ColorSpace(cs.primaries, cs.transfer, "exr:sequence")
    return output, mask, cs
