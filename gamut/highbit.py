"""16-bit RGB/RGBA export with optional TPDF dither before quantisation.

The small PNG and uncompressed TIFF writers use the standard library so core
IO needs no additional imaging dependency. Input is already transfer-encoded;
integer output clips to [0, 1], rounds to the nearest code, and never wraps.
"""

import struct
import zlib
from os import PathLike
from pathlib import Path

import numpy as np
import torch


def tpdf_dither(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Add triangular noise in [-1, 1] output LSBs without clipping or rounding."""
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 1:
        raise ValueError("Dither bit depth must be a positive integer.")
    if not x.is_floating_point():
        raise TypeError("Dither requires floating-point input.")
    noise = (torch.rand_like(x) - torch.rand_like(x)) / (2**bits - 1)
    return x + noise


def _quantize(arr: np.ndarray | torch.Tensor, dither: bool) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(np.ascontiguousarray(arr))
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4) or 0 in arr.shape:
        raise ValueError("16-bit export requires a nonempty [H, W, 3 or 4] image.")
    if not arr.is_floating_point():
        raise TypeError("16-bit export requires floating-point input.")
    # Promote half precision before multiplying by 65535, above half's maximum.
    pixels = arr.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(pixels).all():
        raise ValueError("16-bit export requires finite pixels.")
    if dither:
        pixels = tpdf_dither(pixels, 16)
    return (pixels.clamp(0.0, 1.0) * 65535).round().numpy().astype(np.uint16)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def save_png16(
    path: str | PathLike[str], arr: np.ndarray | torch.Tensor, dither: bool = False
) -> None:
    """Write 16-bit PNG in RGB/RGBA order, with dither disabled by default."""
    pixels = _quantize(arr, dither)
    height, width, channels = pixels.shape
    # PNG uses big-endian samples and a filter byte before every scanline.
    # https://www.w3.org/TR/png-3/#11IHDR
    rows = pixels.astype(">u2").view(np.uint8).reshape(height, -1)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 16, 2 if channels == 3 else 6, 0, 0, 0)
    with Path(path).open("wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.write(_png_chunk(b"IHDR", header))
        output.write(_png_chunk(b"IDAT", zlib.compress(scanlines)))
        output.write(_png_chunk(b"IEND", b""))


def save_tiff16(
    path: str | PathLike[str], arr: np.ndarray | torch.Tensor, dither: bool = False
) -> None:
    """Write a single uncompressed 16-bit TIFF, with unassociated alpha if RGBA."""
    pixels = _quantize(arr, dither)
    height, width, channels = pixels.shape
    # Classic TIFF: little-endian, one IFD and one interleaved pixel strip.
    # https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf
    entry_count = 13 + (channels == 4)
    bits_offset = 8 + 2 + entry_count * 12 + 4
    resolution_offset = bits_offset + channels * 2
    pixels_offset = resolution_offset + 8
    if pixels_offset + pixels.nbytes >= 2**32:
        raise ValueError("Image exceeds the classic TIFF 4 GiB size limit.")
    # Tag, field type (SHORT=3, LONG=4, RATIONAL=5), count, value or offset.
    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, channels, bits_offset),
        (259, 3, 1, 1),       # no compression
        (262, 3, 1, 2),       # RGB photometric interpretation
        (273, 4, 1, pixels_offset),
        (277, 3, 1, channels),
        (278, 4, 1, height),
        (279, 4, 1, pixels.nbytes),
        (282, 5, 1, resolution_offset),
        (283, 5, 1, resolution_offset),
        (284, 3, 1, 1),       # contiguous samples
        (296, 3, 1, 1),       # no absolute resolution unit
    ]
    if channels == 4:
        entries.append((338, 3, 1, 2))  # unassociated alpha
    with Path(path).open("wb") as output:
        output.write(struct.pack("<2sHIH", b"II", 42, 8, entry_count))
        for tag, field_type, count, value in entries:
            output.write(struct.pack("<HHII", tag, field_type, count, value))
        output.write(struct.pack("<I", 0))  # no next IFD
        output.write(struct.pack("<" + "H" * channels, *([16] * channels)))
        output.write(struct.pack("<II", 1, 1))
        output.write(pixels.astype("<u2", copy=False).tobytes())
