"""Real OpenEXR 3.x file round trips, metadata inference, and IO failures."""

import numpy as np
import OpenEXR
import pytest
import torch

from gamut import exr, exr_layers
from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
from gamut.exr import _CANONICAL_NAMES, _PRIMARIES_NEUTRAL, read_exr, write_exr
from gamut.primaries import CHROMATICITIES
from nodes import info_nodes, io_nodes


def _pixels(channels=3):
    return np.linspace(-0.25, 100.0, 2 * 3 * channels, dtype=np.float32).reshape(
        2, 3, channels
    )


# Writable pairs: those we can name truthfully. A transfer whose name is itself
# a complete colour space ("ACEScct" => AP1, "sRGB" => Rec.709) may only be
# written on its own primaries, or the tag would contradict the chromaticities.
WRITABLE = [
    (pri, tr)
    for pri in PRIMARIES
    for tr in TRANSFERS
    if (pri, tr) in _CANONICAL_NAMES or tr in _PRIMARIES_NEUTRAL
]
REFUSED = [
    (pri, tr)
    for pri in PRIMARIES
    for tr in TRANSFERS
    if (pri, tr) not in _CANONICAL_NAMES and tr not in _PRIMARIES_NEUTRAL
]


def _write_raw(path, header, pixels=None):
    if pixels is None:
        pixels = _pixels()
    name = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    with OpenEXR.File(header, {name: pixels}) as image:
        image.write(str(path))


def _layer_file(path, *, alpha=True, header=None, value=2., normal=True, shape=(2, 3)):
    planes = {f"light.diffuse.{component}": np.full(shape, value + i, np.float32)
              for i, component in enumerate("RGB")}
    planes["A"] = np.full(shape, 0.9, np.float32)  # must not leak into a selected layer
    if alpha:
        planes["light.diffuse.A"] = np.full(shape, 0.25, np.float32)
    if normal:
        planes.update({f"normal.{component}": np.full(shape, i - 1., np.float32)
                       for i, component in enumerate("XYZ")})
    with OpenEXR.File(header or {"colorSpace": "ACEScg"}, planes) as file:
        file.write(str(path))


@pytest.mark.parametrize("has_alpha", [False, True])
def test_named_colour_layer_uses_only_its_own_alpha(tmp_path, has_alpha):
    path = tmp_path / "layer.exr"
    _layer_file(path, alpha=has_alpha)
    image, mask, cs = io_nodes.GamutLoadEXR().load(str(path), layer="light.diffuse")
    assert image.shape == (1, 2, 3, 3) and image.dtype == torch.float32
    torch.testing.assert_close(image[0, 0, 0], torch.tensor([2., 3., 4.]))
    assert torch.all(mask == (0.75 if has_alpha else 0.))
    assert cs == ColorSpace("acescg", "linear", "exr:colorSpace")
    with pytest.raises(ValueError, match="light.diffuse"):
        io_nodes.GamutLoadEXR().load(str(path), layer="missing")
    with pytest.raises(ValueError, match="Load Data EXR"):
        io_nodes.GamutLoadEXR().load(str(path), layer="normal", tonemap_preview=True)


def test_colour_reader_keeps_root_rgb_when_auxiliary_depth_exists(tmp_path):
    path = tmp_path / "rgbz.exr"
    with OpenEXR.File({}, {"RGB": _pixels(), "Z": np.zeros((2, 3), np.float32)}) as file:
        file.write(str(path))
    np.testing.assert_array_equal(read_exr(path)[0], _pixels())


@pytest.mark.parametrize("kind", ["colour", "data"])
@pytest.mark.parametrize("pattern", ["shot.{frame}.exr", "shot.####.exr", "shot.%04d.exr"])
def test_selected_sequences_follow_explicit_frame_order(tmp_path, kind, pattern):
    for frame in (7, 9, 11):
        _layer_file(tmp_path / f"shot.{frame:04d}.exr", value=frame)
    node = io_nodes.GamutLoadEXR() if kind == "colour" else io_nodes.GamutLoadDataEXR()
    layer = "light.diffuse" if kind == "colour" else "normal"
    result = node.load(str(tmp_path / pattern), layer=layer, start_frame=7, frame_count=3, frame_step=2)
    assert result[0].shape == (3, 2, 3, 3)
    if kind == "colour":
        torch.testing.assert_close(result[0][:, 0, 0, 0], torch.tensor([7., 9., 11.]))
        assert result[1].shape == (3, 2, 3) and torch.all(result[1] == 0.75)
    else:
        assert len(result) == 1
        torch.testing.assert_close(result[0][:, 0, 0], torch.tensor([[-1., 0., 1.]]).repeat(3, 1))


@pytest.mark.parametrize("fault,match", [
    ("missing", None), ("alpha", "channels"), ("shape", "windows"),
    ("dataWindow", "windows"), ("displayWindow", "windows"), ("space", "colour declaration"),
])
def test_sequence_preflight_refuses_before_tensor_allocation(tmp_path, monkeypatch, fault, match):
    first, second = tmp_path / "shot.1001.exr", tmp_path / "shot.1002.exr"
    _layer_file(first)
    if fault != "missing":
        header = {"colorSpace": "sRGB" if fault == "space" else "ACEScg"}
        if fault == "dataWindow":
            header["dataWindow"] = (np.array([10, 20], np.int32), np.array([12, 21], np.int32))
        if fault == "displayWindow":
            header["displayWindow"] = (np.array([0, 0], np.int32), np.array([99, 99], np.int32))
        if fault == "shape":
            _layer_file(second, header=header, shape=(4, 5))
        else:
            _layer_file(second, alpha=fault != "alpha", header=header)
    def allocate(*args, **kwargs):
        pytest.fail("A bad sequence must be rejected before allocating its tensor batch")
    monkeypatch.setattr(exr.torch, "empty", allocate)
    with pytest.raises((ValueError, RuntimeError, OSError), match=match):
        io_nodes.GamutLoadEXR().load(str(tmp_path / "shot.####.exr"), layer="light.diffuse",
                                   frame_count=2, tonemap_preview=True)


def test_sequence_accepts_fixed_nonzero_window_and_records_mixed_provenance(tmp_path):
    window = (np.array([10, 20], np.int32), np.array([12, 21], np.int32))
    shared = {"dataWindow": window, "displayWindow": window}
    _layer_file(tmp_path / "shot.1001.exr", header={**shared, "colorSpace": "ACEScg"})
    _layer_file(tmp_path / "shot.1002.exr", header={**shared, "chromaticities": tuple(CHROMATICITIES["acescg"])})
    result = io_nodes.GamutLoadEXR().load(str(tmp_path / "shot.####.exr"), layer="light.diffuse", frame_count=2)
    assert result[0].shape == (2, 2, 3, 3)
    assert result[2] == ColorSpace("acescg", "linear", "exr:sequence")


@pytest.mark.parametrize("node", [io_nodes.GamutLoadEXR, io_nodes.GamutLoadDataEXR])
def test_later_frame_fingerprint_and_missing_marker(tmp_path, node):
    path = tmp_path / "shot.####.exr"
    _layer_file(tmp_path / "shot.1001.exr")
    kwargs = dict(path=str(path), frame_count=2)
    missing = node.IS_CHANGED(**kwargs)
    assert missing[-1][-1] == "missing"
    _layer_file(tmp_path / "shot.1002.exr")
    present = node.IS_CHANGED(**kwargs)
    assert present != missing
    # File contents, not only first-frame mtime, must matter.
    with (tmp_path / "shot.1002.exr").open("ab") as file:
        file.write(b"changed")
    assert node.IS_CHANGED(**kwargs) != present


@pytest.mark.parametrize("roles", [
    "not json", "[]", '{"schema":2,"colour":[],"data":[]}',
    '{"schema":true,"colour":[],"data":[]}',
    '{"schema":1,"colour":["rgba"],"data":["rgba"]}',
    '{"schema":1,"colour":["rgba","rgba"],"data":[]}',
    '{"schema":1,"colour":["missing"],"data":[]}',
    '{"schema":1,"colour":[],"data":[]}',
    '{"schema":1,"colour":"rgba","data":[]}',
])
def test_malformed_layer_roles_are_not_silently_ignored(tmp_path, roles):
    path = tmp_path / "bad.exr"
    _write_raw(path, {"gamut:layerRoles": roles})
    with pytest.raises(ValueError, match="gamut:layerRoles"):
        read_exr(path)


def test_role_marked_rgb_data_refuses_colour_but_loads_raw(tmp_path):
    path = tmp_path / "raw.exr"
    _write_raw(path, {"colorSpace": "ACEScg", "gamut:layerRoles":
                    '{"schema":1,"colour":[],"data":["rgba"]}'})
    with pytest.raises(ValueError, match="Load Data EXR"):
        read_exr(path)
    image, = io_nodes.GamutLoadDataEXR().load(str(path), layer="", components="RGB")
    np.testing.assert_array_equal(image.numpy()[0], _pixels())


def test_info_reports_data_layers_and_types_without_decoding_pixels(tmp_path, monkeypatch):
    path = tmp_path / "named.exr"
    _layer_file(path)
    original = OpenEXR.File
    def header_only(*args, **kwargs):
        assert kwargs.get("header_only") is True
        return original(*args, **kwargs)
    monkeypatch.setattr(OpenEXR, "File", header_only)
    text = info_nodes.GamutColorSpaceInfo().inspect(path=str(path))["result"][0]
    assert "light.diffuse" in text and "normal" in text and "X (FLOAT)" in text


@pytest.mark.parametrize("kind", ["deep", "multipart", "subsampled"])
@pytest.mark.parametrize("reader", [io_nodes.GamutLoadEXR, io_nodes.GamutLoadDataEXR])
def test_unsupported_storage_is_refused_from_real_headers(tmp_path, kind, reader):
    path = tmp_path / "unsupported.exr"
    if kind == "deep":
        pixels = np.empty((2, 4), dtype=object)
        for index in np.ndindex(pixels.shape):
            pixels[index] = np.array([1.], np.float32)
        file = OpenEXR.File({"type": OpenEXR.deepscanline, "compression": OpenEXR.ZIPS_COMPRESSION},
                            {c: pixels for c in "RGB"})
    elif kind == "multipart":
        file = OpenEXR.File([OpenEXR.Part({"name": n}, {"RGB": _pixels()}) for n in ("one", "two")])
    else:
        file = OpenEXR.File({}, {c: OpenEXR.Channel(np.ones((2, 4), np.float32), 2, 1) for c in "RGB"})
    with file:
        file.write(str(path))
    kwargs = {"layer": "", "components": "RGB"} if reader is io_nodes.GamutLoadDataEXR else {}
    with pytest.raises(ValueError, match={"deep": "deep", "multipart": "single-part", "subsampled": "Subsampled"}[kind]):
        reader().load(str(path), **kwargs)


@pytest.mark.parametrize("primaries,transfer", WRITABLE)
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


@pytest.mark.parametrize("primaries,transfer", REFUSED)
def test_write_refuses_tag_that_would_contradict_the_chromaticities(
    tmp_path, primaries, transfer
):
    """A complete-space transfer name on foreign primaries must not be written.

    ACEScct is defined on AP1 and sRGB on Rec.709. Writing either name beside
    different chromaticities produces a file whose tag and header disagree, and
    a consumer that trusts the name applies the wrong gamut.
    """
    with pytest.raises(ValueError, match="contradict the chromaticities"):
        write_exr(
            tmp_path / "bad.exr",
            _pixels(),
            ColorSpace(primaries, transfer, "widget"),
        )


def test_half_write_refuses_values_that_would_become_inf(tmp_path):
    """Finite highlights must never be silently turned into inf by the cast."""
    hot = np.full((2, 2, 3), 112802.9375, dtype=np.float32)  # LTX 1.0 at +11 stops
    cs = ColorSpace("rec709", "linear", "widget")
    with pytest.raises(ValueError, match="exceeds the float16 maximum"):
        write_exr(tmp_path / "hot.exr", hot, cs, half=True)
    # float32 carries it fine.
    write_exr(tmp_path / "hot32.exr", hot, cs, half=False)
    pixels, _, _ = read_exr(tmp_path / "hot32.exr")
    assert np.isfinite(pixels).all()
    np.testing.assert_allclose(pixels.max(), 112802.9375, rtol=0, atol=0)


def test_arri_camera_logc3_name_is_not_claimed_as_rec709(tmp_path):
    """ARRI LogC3 EI800 is AWG3, not Rec.709; we must refuse, not misidentify."""
    path = tmp_path / "arri.exr"
    _write_raw(path, {"colorSpace": "ARRI LogC3 (EI800)"})
    _, inferred, _ = read_exr(path)
    assert inferred is None


# --- Independent golden fixtures ------------------------------------------
# Hand-written, NOT derived from gamut.exr's registries. If someone edits
# _CANONICAL_NAMES the round-trip tests above would follow the edit silently;
# these will not. The strings are the names Nuke and the OCIO ACES config
# actually use, so this table is an interoperability contract, not a restating
# of the implementation.
GOLDEN_TAGS = {
    ("acescg", "linear"): "ACEScg",
    ("ap0", "linear"): "ACES2065-1",
    ("rec709", "linear"): "Linear Rec.709 (sRGB)",
    ("rec2020", "linear"): "Linear Rec.2020",
    ("rec709", "srgb"): "sRGB",
    ("acescg", "acescct"): "ACEScct",
    ("rec709", "logc3"): "LogC3",
}

GOLDEN_CHROMATICITIES = {
    "rec709": (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290),
    "acescg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "ap0": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
    "rec2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
}


@pytest.mark.parametrize("pair,expected_tag", sorted(GOLDEN_TAGS.items()))
def test_written_tag_matches_hand_written_golden_name(tmp_path, pair, expected_tag):
    primaries, transfer = pair
    path = tmp_path / "golden.exr"
    write_exr(path, _pixels(), ColorSpace(primaries, transfer, "widget"), half=False)
    with OpenEXR.File(str(path)) as image:
        assert image.header()["colorSpace"] == expected_tag


@pytest.mark.parametrize("primaries,expected", sorted(GOLDEN_CHROMATICITIES.items()))
def test_written_chromaticities_match_hand_written_golden_values(
    tmp_path, primaries, expected
):
    path = tmp_path / "golden_chroma.exr"
    write_exr(path, _pixels(), ColorSpace(primaries, "linear", "widget"), half=False)
    with OpenEXR.File(str(path)) as image:
        np.testing.assert_allclose(
            np.asarray(image.header()["chromaticities"], dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0,
            atol=1e-6,
        )
