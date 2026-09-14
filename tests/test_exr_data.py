"""Untagged data-channel EXR IO: round trips, refusals, and the absence invariant.

The load-bearing property here is a *negative* one — a data file must carry no
`chromaticities` and no `colorSpace` — because that absence is how a reader
tells a data file from a tagged colour file. Several tests below assert on what
is missing rather than what is present, and they are the point of the file.

The colour-path regression tests at the bottom exist because both writers now
share `_guard_half_range`; a change to that helper must not alter the colour
path's behaviour or its error text.
"""

import numpy as np
import OpenEXR
import pytest

from gamut.colorspace import ColorSpace
from gamut.exr import read_data_exr, read_exr, write_data_exr, write_exr
from gamut.exr_layers import write_multilayer_exr
from nodes.io_nodes import GamutLoadDataEXR


def _channel(h=2, w=3, start=-0.25, stop=4.0):
    return np.linspace(start, stop, h * w, dtype=np.float32).reshape(h, w)


def _normals(h=2, w=3):
    return {
        "N.X": _channel(h, w, -1.0, 1.0),
        "N.Y": _channel(h, w, 1.0, -1.0),
        "N.Z": _channel(h, w, 0.0, 1.0),
    }


def test_named_channels_round_trip(tmp_path):
    """Arbitrary channel names survive a write/read cycle with their pixels."""
    path = tmp_path / "normals.exr"
    written = _normals()
    write_data_exr(path, written, half=False)
    read, _ = read_data_exr(path)
    assert set(read) == set(written)
    for name, pixels in written.items():
        assert np.allclose(read[name], pixels, rtol=0, atol=1e-6)


def test_mixed_file_data_reader_never_applies_its_colour_declaration(tmp_path):
    path = tmp_path / "mixed.exr"
    vectors = np.stack([_channel(start=-1, stop=1)] * 3, axis=-1)
    write_multilayer_exr(path, {"rgb": vectors, "normal": vectors},
                         ColorSpace("rec709", "srgb", "test"))
    channels, header = read_data_exr(path)
    assert header["colorSpace"] == "sRGB"
    np.testing.assert_array_equal(channels["normal.X"], vectors[..., 0])
    image, = GamutLoadDataEXR().load(str(path))
    np.testing.assert_array_equal(image.numpy()[0], vectors)


def test_old_n_xyz_exports_load_with_data_node(tmp_path):
    path = tmp_path / "old.exr"
    write_data_exr(path, _normals(), half=False)
    image, = GamutLoadDataEXR().load(str(path), layer="N")
    for index, name in enumerate(("N.X", "N.Y", "N.Z")):
        np.testing.assert_array_equal(image.numpy()[0, ..., index], _normals()[name])


def test_written_data_file_declares_no_colour(tmp_path):
    """The invariant: neither attribute is present, at the raw-header level."""
    path = tmp_path / "normals.exr"
    write_data_exr(path, _normals(), half=False)
    _, header = read_data_exr(path)
    assert "chromaticities" not in header
    assert "colorSpace" not in header


def test_data_reader_never_infers_colour(tmp_path):
    """RGB-named channels are returned as data, with no colour interpretation.

    A data file may legitimately use the names R, G and B. The data reader must
    hand them back untouched rather than treating them as an image.
    """
    path = tmp_path / "rgbnamed.exr"
    write_data_exr(path, {"R": _channel(), "G": _channel(), "B": _channel()}, half=False)
    channels, header = read_data_exr(path)
    assert set(channels) == {"R", "G", "B"}
    assert "chromaticities" not in header and "colorSpace" not in header


def test_data_writer_accepts_no_colourspace():
    """It must be impossible to hand this writer a colour declaration."""
    with pytest.raises(TypeError):
        write_data_exr(  # type: ignore[call-arg]
            "unused.exr", _normals(), colorspace=ColorSpace("acescg", "linear", "test")
        )


def test_single_channel_is_valid(tmp_path):
    """A depth or coverage pass is one channel; that is not a degenerate case."""
    path = tmp_path / "depth.exr"
    write_data_exr(path, {"Z": _channel(start=0.1, stop=1000.0)}, half=False)
    channels, _ = read_data_exr(path)
    assert list(channels) == ["Z"]


def test_mismatched_channel_shapes_are_refused(tmp_path):
    """Channels of differing size cannot form one image."""
    with pytest.raises(ValueError, match="share one shape"):
        write_data_exr(
            tmp_path / "bad.exr", {"A": _channel(2, 3), "B": _channel(3, 4)}, half=False
        )


@pytest.mark.parametrize(
    "channels,match",
    [
        ({}, "at least one channel"),
        ({"A": np.zeros((2, 3), dtype=np.int32)}, "floating-point"),
        ({"A": np.zeros((2, 3, 3), dtype=np.float32)}, r"\[H, W\]"),
        ({"A": np.zeros((0, 3), dtype=np.float32)}, r"\[H, W\]"),
        ({"": np.zeros((2, 3), dtype=np.float32)}, "channel names"),
    ],
)
def test_invalid_input_is_refused(tmp_path, channels, match):
    """Empty mappings, integer pixels, wrong rank, empty images, blank names."""
    with pytest.raises((ValueError, TypeError), match=match):
        write_data_exr(tmp_path / "bad.exr", channels, half=False)


def test_non_mapping_is_refused(tmp_path):
    """A list of arrays has no channel names, so it cannot be written."""
    with pytest.raises(TypeError, match="mapping"):
        write_data_exr(tmp_path / "bad.exr", [_channel()], half=False)  # type: ignore[arg-type]


def test_unknown_compression_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown EXR compression"):
        write_data_exr(tmp_path / "bad.exr", _normals(), compression="brotli")


def test_half_guard_fires_on_the_data_path_and_names_the_channel(tmp_path):
    """The overflow guard must protect data channels, not just colour.

    Depth in millimetres or a world-space position trivially exceeds float16
    range, and silently writing inf would be the worst possible outcome for a
    geometry pass. The channel name is in the message because a data file may
    carry many channels and only one may overflow.
    """
    channels = {"N.X": _channel(), "Z": np.full((2, 3), 70000.0, dtype=np.float32)}
    with pytest.raises(ValueError, match="exceeds the float16 maximum") as exc:
        write_data_exr(tmp_path / "over.exr", channels, half=True)
    assert "Z" in str(exc.value)
    assert "N.X" not in str(exc.value)


def test_half_guard_ignores_non_finite_values(tmp_path):
    """inf and nan already in the data are not what the guard is for."""
    channels = {"A": np.array([[np.inf, np.nan, 1.0], [0.0, -np.inf, 2.0]], dtype=np.float32)}
    write_data_exr(tmp_path / "nonfinite.exr", channels, half=True)


def test_half_write_is_half_precision(tmp_path):
    """half=True must actually reduce precision, or the guard guards nothing."""
    path = tmp_path / "half.exr"
    write_data_exr(path, {"A": np.full((2, 3), 1.0 / 3.0, dtype=np.float32)}, half=True)
    channels, _ = read_data_exr(path)
    assert channels["A"][0, 0] != np.float32(1.0 / 3.0)
    assert abs(float(channels["A"][0, 0]) - 1.0 / 3.0) < 1e-3


# Deliberately not covered here: the reader's multipart and deep-file
# refusals. Constructing either requires writing a file this module cannot
# produce, and a test that only round-trips a flat file while claiming to
# cover deep data would assert nothing. Recorded in coverage-debt.md instead.


# --- colour-path regression, because the two writers now share a helper ------


def test_colour_path_still_writes_its_declaration(tmp_path):
    """write_exr must be unaffected by the data path's arrival."""
    path = tmp_path / "colour.exr"
    pixels = np.stack([_channel(), _channel(), _channel()], axis=-1)
    write_exr(path, pixels, ColorSpace("acescg", "linear", "test"), half=False)
    _, declared, header = read_exr(path)
    assert "chromaticities" in header
    assert header["colorSpace"] == "ACEScg"
    assert declared is not None and declared.primaries == "acescg"


def test_colour_path_half_guard_message_is_unchanged(tmp_path):
    """The shared helper must keep the phrase the existing suite matches on."""
    pixels = np.full((2, 3, 3), 70000.0, dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds the float16 maximum"):
        write_exr(
            tmp_path / "over.exr", pixels, ColorSpace("acescg", "linear", "test"), half=True
        )


def test_data_and_colour_files_are_distinguishable(tmp_path):
    """End to end: the absence invariant actually separates the two kinds.

    This is the property every downstream consumer relies on, so it gets its
    own test rather than living implicitly inside the others.
    """
    data_path = tmp_path / "data.exr"
    colour_path = tmp_path / "colour.exr"
    write_data_exr(data_path, _normals(), half=False)
    pixels = np.stack([_channel(), _channel(), _channel()], axis=-1)
    write_exr(colour_path, pixels, ColorSpace("acescg", "linear", "test"), half=False)

    def has_colour_metadata(path):
        with OpenEXR.File(str(path), separate_channels=True) as image:
            header = dict(image.header())
        return "chromaticities" in header or "colorSpace" in header

    assert not has_colour_metadata(data_path)
    assert has_colour_metadata(colour_path)
