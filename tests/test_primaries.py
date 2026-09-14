"""Independent colour-science checks, including the fabricated-matrix regression."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib.util import find_spec
from itertools import permutations, product
from pathlib import Path

import pytest
import torch

from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace, describe
from gamut.primaries import CHROMATICITIES, MATRICES, convert


# Independent published xy coordinates, never read from production constants.
# Rec.709 / BT.2020: ITU-R; AP0 / AP1 and ACES white: SMPTE ST 2065-1 / AMPAS.
REFERENCE_XY = {
    "rec709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), (0.3127, 0.3290)),
    "acescg": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), (0.32168, 0.33767)),
    "ap0": ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.077), (0.32168, 0.33767)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), (0.3127, 0.3290)),
}
PAIRS = list(permutations(REFERENCE_XY, 2))
from .conftest import DEVICES  # probed, not just is_available()
HAS_OCIO = find_spec("PyOpenColorIO") is not None


def _rgb_to_xyz_and_white(name):
    xy = torch.tensor(REFERENCE_XY[name], dtype=torch.float64)
    # xyY with Y=1 becomes (x/y, 1, (1-x-y)/y).
    xyz = torch.stack(
        (
            xy[:, 0] / xy[:, 1],
            torch.ones(4, dtype=torch.float64),
            (1.0 - xy[:, 0] - xy[:, 1]) / xy[:, 1],
        ),
        dim=1,
    )
    unscaled = xyz[:3].T
    white = xyz[3]
    scales = torch.linalg.solve(unscaled, white)
    return unscaled @ torch.diag(scales), white


def _derive_matrix(src, dst):
    source, source_white = _rgb_to_xyz_and_white(src)
    destination, destination_white = _rgb_to_xyz_and_white(dst)
    bradford = torch.tensor(
        (
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        ),
        dtype=torch.float64,
    )
    cone_scale = (bradford @ destination_white) / (bradford @ source_white)
    adaptation = torch.linalg.solve(bradford, torch.diag(cone_scale) @ bradford)
    return torch.linalg.solve(destination, adaptation @ source)


def test_every_selectable_ordered_pair_has_an_explicit_matrix():
    assert set(PRIMARIES) == set(REFERENCE_XY)
    assert set(MATRICES) == {f"{src}->{dst}" for src, dst in PAIRS}
    assert len(MATRICES) == 12


@pytest.mark.parametrize("name", REFERENCE_XY)
def test_header_chromaticities_match_independent_reference(name):
    expected = torch.tensor(REFERENCE_XY[name], dtype=torch.float64).flatten()
    actual = torch.tensor(CHROMATICITIES[name], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("src,dst", PAIRS)
def test_matrix_matches_first_principles_primaries_and_bradford(src, dst):
    actual = torch.tensor(MATRICES[f"{src}->{dst}"], dtype=torch.float64)
    torch.testing.assert_close(actual, _derive_matrix(src, dst), atol=1e-6, rtol=0)


@pytest.mark.parametrize("src,dst", PAIRS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_pair_converts_basis_vectors_and_round_trips(src, dst, dtype):
    basis = torch.eye(3, dtype=dtype)
    expected = _derive_matrix(src, dst).T.to(dtype=dtype)
    torch.testing.assert_close(convert(basis, src, dst), expected, atol=1e-6, rtol=0)
    img = torch.linspace(-0.25, 2.0, 72, dtype=dtype).reshape(2, 3, 4, 3)
    result = convert(convert(img, src, dst), dst, src)
    torch.testing.assert_close(result, img, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "src,dst",
    [
        (src, dst) for src, dst in PAIRS
        if REFERENCE_XY[src][-1] == REFERENCE_XY[dst][-1]
    ],
)
def test_same_white_point_matrix_rows_sum_to_one(src, dst):
    matrix = torch.tensor(MATRICES[f"{src}->{dst}"], dtype=torch.float64)
    torch.testing.assert_close(
        matrix.sum(dim=1), torch.ones(3, dtype=torch.float64), atol=1e-6, rtol=0
    )


def test_rec709_to_acescg_is_not_fabricated_06724_01536_01740_matrix():
    fabricated = torch.tensor(
        (
            (0.6724, 0.1536, 0.1740),
            (0.0054, 1.0070, -0.0123),
            (-0.0823, -0.0386, 1.1209),
        ),
        dtype=torch.float64,
    )
    actual = torch.tensor(MATRICES["rec709->acescg"], dtype=torch.float64)
    assert not torch.allclose(actual, fabricated, atol=1e-6, rtol=0)


@pytest.mark.parametrize("src,dst", list(product(REFERENCE_XY, repeat=2)))
@pytest.mark.parametrize("transfer", [name for name in TRANSFERS if name != "linear"])
def test_convert_rejects_non_linear_transfer_even_for_identity(src, dst, transfer):
    img = torch.ones(3)
    cs = ColorSpace(src, transfer, "widget")
    with pytest.raises(ValueError) as exc:
        convert(img, src, dst, colorspace=cs)
    message = str(exc.value)
    assert transfer in message
    assert "Gamut: Transfer" in message
    assert "decode" in message


def test_untagged_input_is_assumed_linear():
    img = torch.tensor([0.1, 0.2, 2.0])
    cs = ColorSpace("rec709", "linear", "exr:chromaticities")
    torch.testing.assert_close(
        convert(img, "rec709", "acescg"),
        convert(img, "rec709", "acescg", colorspace=cs),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize("name", REFERENCE_XY)
def test_identity_returns_the_original_tensor(name):
    img = torch.tensor([-0.1, 0.2, 2.0, 0.5])
    assert convert(img, name, name) is img


@pytest.mark.parametrize("shape", [(3,), (4,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_last_axis_any_rank_and_extra_channels_pass_through(shape):
    img = torch.linspace(-0.5, 2.0, torch.Size(shape).numel(), dtype=torch.float64)
    img = img.reshape(shape)
    original = img.clone()
    expected = img[..., :3] @ _derive_matrix("ap0", "rec709").T
    actual = convert(img, "ap0", "rec709")
    assert actual.shape == img.shape
    torch.testing.assert_close(actual[..., :3], expected, atol=1e-6, rtol=0)
    assert torch.equal(actual[..., 3:], img[..., 3:])
    assert torch.equal(img, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_working_dtype_and_device_are_preserved(dtype, device):
    img = torch.eye(3, dtype=dtype, device=device)
    actual = convert(img, "rec709", "acescg")
    assert actual.dtype == img.dtype
    assert actual.device == img.device
    expected = _derive_matrix("rec709", "acescg").T.to(dtype=dtype, device=device)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_noncontiguous_input_and_negative_hdr_output():
    img = torch.eye(3, dtype=torch.float64).T
    assert not img.is_contiguous()
    actual = convert(img, "acescg", "rec709")
    torch.testing.assert_close(
        actual, _derive_matrix("acescg", "rec709").T, atol=1e-6, rtol=0
    )
    assert actual.min() < 0
    assert actual.max() > 1


@pytest.mark.parametrize("shape", [(), (2,), (1, 2, 2), (1, 0)])
@pytest.mark.parametrize("dst", ["rec709", "acescg"])
def test_fewer_than_three_channels_is_rejected(shape, dst):
    with pytest.raises(ValueError, match="C >= 3"):
        convert(torch.zeros(shape), "rec709", dst)


@pytest.mark.parametrize(
    "src,dst", [("unknown", "acescg"), ("rec709", "unknown"), ("unknown", "unknown")]
)
def test_unknown_pair_error_names_both_spaces(src, dst):
    with pytest.raises(KeyError) as exc:
        convert(torch.ones(3), src, dst)
    assert src in str(exc.value)
    assert dst in str(exc.value)


def test_colorspace_is_frozen_and_description_records_provenance():
    cs = ColorSpace("acescg", "linear", "exr:chromaticities")
    with pytest.raises(FrozenInstanceError):
        cs.transfer = "srgb"
    description = describe(cs)
    assert PRIMARIES[cs.primaries] in description
    assert TRANSFERS[cs.transfer] in description
    assert cs.source in description
    assert description != describe(ColorSpace("acescg", "linear", "widget"))
    assert "custom-space" in describe(ColorSpace("custom-space", "linear", "ocio:custom"))


@pytest.mark.parametrize("block_torch", [False, True])
def test_core_imports_without_optional_ocio_and_declarations_without_torch(block_torch):
    script = """
import importlib.abc
import sys

class BlockDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ['PyOpenColorIO', 'opencolorio', 'gamut.ocio_bridge']
        if sys.argv[1] == 'True':
            blocked.append('torch')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('Dependency deliberately absent: ' + fullname)
        return None

sys.meta_path.insert(0, BlockDependencies())
import gamut
import gamut.colorspace
if sys.argv[1] == 'False':
    import gamut.primaries
    import gamut.transfer
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(block_torch)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not HAS_OCIO, reason="Optional opencolorio is not installed")
@pytest.mark.parametrize("src,dst", PAIRS)
def test_matrices_match_ocio_when_installed(src, dst):
    import PyOpenColorIO as ocio

    config = ocio.Config.CreateFromBuiltinConfig("cg-config-v4.0.0_aces-v2.0_ocio-v2.5")
    names = {
        "rec709": "Linear Rec.709 (sRGB)",
        "acescg": "ACEScg",
        "ap0": "ACES2065-1",
        "rec2020": "Linear Rec.2020",
    }
    processor = config.getProcessor(names[src], names[dst]).getDefaultCPUProcessor()
    basis = torch.eye(3, dtype=torch.float32)
    expected = torch.tensor([processor.applyRGB(rgb.tolist()) for rgb in basis])
    torch.testing.assert_close(convert(basis, src, dst), expected, atol=1e-5, rtol=0)
