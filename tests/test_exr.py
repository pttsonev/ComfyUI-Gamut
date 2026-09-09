"""Real OpenEXR 3.x file round trips, metadata inference, and IO failures."""

import numpy as np
import OpenEXR
import pytest

from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
from gamut.exr import _CANONICAL_NAMES, read_exr, write_exr
from gamut.primaries import CHROMATICITIES


def _pixels(channels=3):
    return np.linspace(-0.25, 100.0, 2 * 3 * channels, dtype=np.float32).reshape(
        2, 3, channels
    )


def _write_raw(path, header, pixels=None):
    if pixels is None:
        pixels = _pixels()
    name = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    with OpenEXR.File(header, {name: pixels}) as image:
        image.write(str(path))


@pytest.mark.parametrize("primaries", PRIMARIES)
@pytest.mark.parametrize("transfer", TRANSFERS)
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("channels", [3, 4])
def test_pixels_and_both_real_header_tags_round_trip(
    tmp_path, primaries, transfer, half, channels
):
    path = tmp_path / "tagged.exr"
    original = _pixels(channels)
    cs = ColorSpace(primaries, transfer, "widget")
    write_exr(path, original, cs, half=half)
    pixels, inferred, header = read_exr(path)
    expected = original.astype(np.float16).astype(np.float32) if half else original
    np.testing.assert_array_equal(pixels, expected)
    assert pixels.dtype == np.float32
    assert pixels.flags.c_contiguous
    assert inferred == ColorSpace(primaries, transfer, "exr:chromaticities")
    assert header["colorSpace"] == _CANONICAL_NAMES.get(
        (primaries, transfer), TRANSFERS[transfer]
    )
    np.testing.assert_allclose(
        header["chromaticities"], CHROMATICITIES[primaries], atol=1e-6, rtol=0
    )
    # Read through the binding independently, not just through our reader.
    with OpenEXR.File(str(path)) as image:
        raw = image.channels()["RGBA" if channels == 4 else "RGB"].pixels
        assert raw.dtype == (np.float16 if half else np.float32)
        assert image.header()["colorSpace"] == _CANONICAL_NAMES.get(
            (primaries, transfer), TRANSFERS[transfer]
        )
        assert len(image.header()["chromaticities"]) == 8
        np.testing.assert_allclose(
            image.header()["chromaticities"], CHROMATICITIES[primaries],
            atol=1e-6, rtol=0,
        )
    assert pixels.min() < 0
    assert pixels.max() > 1


@pytest.mark.parametrize(
    "compression,expected",
    [("none", OpenEXR.NO_COMPRESSION), ("rle", OpenEXR.RLE_COMPRESSION),
     ("zips", OpenEXR.ZIPS_COMPRESSION), ("zip", OpenEXR.ZIP_COMPRESSION),
     ("piz", OpenEXR.PIZ_COMPRESSION)],
)
def test_lossless_compression_and_noncontiguous_input(tmp_path, compression, expected):
    path = tmp_path / "compressed.exr"
    pixels = _pixels(4).transpose(1, 0, 2)
    original = pixels.copy()
    write_exr(path, pixels, ColorSpace("rec709", "linear", "widget"),
              half=False, compression=compression)
    actual, _, header = read_exr(path)
    assert header["compression"] == expected
    np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(pixels, original)


def test_default_is_half_float_zip(tmp_path):
    path = tmp_path / "default.exr"
    write_exr(path, _pixels(), ColorSpace("rec709", "linear", "widget"))
    with OpenEXR.File(str(path)) as image:
        assert image.channels()["RGB"].pixels.dtype == np.float16
        assert image.header()["compression"] == OpenEXR.ZIP_COMPRESSION


def test_untagged_file_returns_none_and_preserves_raw_header(tmp_path):
    path = tmp_path / "untagged.exr"
    pixels = _pixels(4)
    _write_raw(path, {"owner": "test writer"}, pixels)
    actual, cs, header = read_exr(path)
    np.testing.assert_array_equal(actual, pixels)
    assert cs is None
    assert header["owner"] == "test writer"
    assert "colorSpace" not in header
    assert "chromaticities" not in header


@pytest.mark.parametrize("delta,known", [(0.00005, True), (0.0005, False)])
def test_chromaticities_tolerance_and_unknown_values_retained(tmp_path, delta, known):
    path = tmp_path / "chroma.exr"
    chroma = tuple(value + delta for value in CHROMATICITIES["rec709"])
    _write_raw(path, {"chromaticities": chroma, "colorSpace": "Linear"})
    _, cs, header = read_exr(path)
    if known:
        assert cs == ColorSpace("rec709", "linear", "exr:chromaticities")
    else:
        assert cs is None
    np.testing.assert_allclose(header["chromaticities"], chroma, atol=1e-6, rtol=0)


def test_chromaticities_only_uses_linear_exr_convention(tmp_path):
    path = tmp_path / "chroma_only.exr"
    _write_raw(path, {"chromaticities": CHROMATICITIES["acescg"]})
    assert read_exr(path)[1] == ColorSpace("acescg", "linear", "exr:chromaticities")


@pytest.mark.parametrize(
    "tag,primaries,transfer",
    [("ACEScg", "acescg", "linear"), ("ACES2065-1", "ap0", "linear"),
     ("Linear Rec.709 (sRGB)", "rec709", "linear"),
     ("Linear Rec.2020", "rec2020", "linear"), ("sRGB", "rec709", "srgb"),
     ("ACEScct", "acescg", "acescct")],
)
def test_complete_colorspace_name_without_chromaticities(tmp_path, tag, primaries, transfer):
    path = tmp_path / "name_only.exr"
    _write_raw(path, {"colorSpace": tag})
    assert read_exr(path)[1] == ColorSpace(primaries, transfer, "exr:colorSpace")


@pytest.mark.parametrize(
    "header",
    [{"colorSpace": "Linear"}, {"colorSpace": "unknown space"},
     {"chromaticities": CHROMATICITIES["rec709"], "colorSpace": "unknown curve"},
     {"chromaticities": CHROMATICITIES["rec709"], "colorSpace": "ACEScg"},
     {"chromaticities": (0.1,) * 8, "colorSpace": "ACEScg"}],
)
def test_incomplete_unknown_or_conflicting_tags_do_not_invent_a_space(tmp_path, header):
    path = tmp_path / "unknown.exr"
    _write_raw(path, header)
    assert read_exr(path)[1] is None


def test_offset_data_window_does_not_shift_pixels(tmp_path):
    path = tmp_path / "offset.exr"
    header = {"dataWindow": (np.array([10, 20], np.int32), np.array([12, 21], np.int32))}
    _write_raw(path, header)
    pixels, _, raw = read_exr(path)
    np.testing.assert_array_equal(pixels, _pixels())
    np.testing.assert_array_equal(raw["dataWindow"][0], [10, 20])


def test_non_rgb_and_multipart_files_are_rejected(tmp_path):
    path = tmp_path / "depth.exr"
    with OpenEXR.File({}, {"Z": np.zeros((2, 3), dtype=np.float32)}) as image:
        image.write(str(path))
    with pytest.raises(ValueError, match="R, G, and B"):
        read_exr(path)
    parts = [OpenEXR.Part({"name": name}, {"RGB": _pixels()}) for name in ("left", "right")]
    with OpenEXR.File(parts) as image:
        image.write(str(path))
    with pytest.raises(ValueError, match="single-part"):
        read_exr(path)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 2, 3, 3), (2, 3, 2), (0, 3, 3)])
def test_invalid_shapes_are_rejected_before_writing(tmp_path, shape):
    path = tmp_path / "invalid.exr"
    with pytest.raises(ValueError, match="nonempty"):
        write_exr(path, np.zeros(shape), ColorSpace("rec709", "linear", "widget"))
    assert not path.exists()


@pytest.mark.parametrize("field", ["primaries", "transfer", "compression", "dtype"])
def test_invalid_write_parameters_fail_loudly(tmp_path, field):
    cs = ColorSpace("unknown" if field == "primaries" else "rec709",
                    "unknown" if field == "transfer" else "linear", "widget")
    pixels = _pixels().astype(np.uint16) if field == "dtype" else _pixels()
    with pytest.raises((ValueError, TypeError)):
        write_exr(tmp_path / "invalid.exr", pixels, cs,
                  compression="unknown" if field == "compression" else "zip")


def test_file_errors_propagate(tmp_path):
    with pytest.raises((RuntimeError, OSError)):
        read_exr(tmp_path / "missing.exr")
    with pytest.raises((RuntimeError, OSError)):
        write_exr(tmp_path / "missing" / "out.exr", _pixels(),
                  ColorSpace("rec709", "linear", "widget"))
