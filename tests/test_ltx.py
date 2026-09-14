"""LTX's bounded VAE input, ordinary LogC3 codes, and unbounded HDR output."""

import numpy as np
import pytest
import torch

from gamut import transfer
from gamut.colorspace import PRIMARIES, ColorSpace
from gamut.exr import read_exr, write_exr
from gamut.ltx import decode_ltx_hdr
from gamut.primaries import MATRICES
from tests.conftest import DEVICES


def _upstream_decode(codes):
    codes = codes.clamp(0.0, 1.0)
    return torch.where(
        codes > 5.367655 * 0.010591 + 0.092809,
        (torch.pow(10.0, (codes - 0.385537) / 0.247190) - 0.052272) / 5.555556,
        (codes - 0.092809) / 5.367655,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("clamp_negatives", [False, True])
def test_known_vae_ramp_matches_upstream_formula(device, dtype, clamp_negatives):
    img = torch.linspace(0, 1, 21, dtype=dtype, device=device).reshape(1, 1, 7, 3)
    original = img.clone()
    actual, cs = decode_ltx_hdr(img, clamp_negatives=clamp_negatives)
    expected = _upstream_decode(img)
    if clamp_negatives:
        expected = expected.clamp_min(0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    assert cs == ColorSpace("rec709", "linear", "node:GamutLTXHDRDecode")
    assert (actual.shape, actual.dtype, actual.device) == (img.shape, img.dtype, img.device)
    assert torch.equal(img, original)


def test_vae_input_is_clamped_and_half_code_is_not_rescaled(monkeypatch):
    img = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
    received = []
    original_decode = transfer.logc3_decode

    def record_codes(codes):
        received.append(codes.clone())
        return original_decode(codes)

    monkeypatch.setattr(transfer, "logc3_decode", record_codes)
    actual, _ = decode_ltx_hdr(img, clamp_negatives=False)
    assert len(received) == 1
    torch.testing.assert_close(received[0], torch.tensor([0., 0.5, 1.], dtype=img.dtype))
    assert actual[1].item() == pytest.approx(0.513383, abs=1e-6)
    assert actual[0].item() == pytest.approx(-0.017290, abs=1e-6)
    assert actual[2].item() == pytest.approx(55.079577, abs=1e-6)
    # The generic curve remains unbounded; only the VAE adapter clamps codes.
    assert original_decode(torch.tensor(1.2)).item() > actual[2].item()


@pytest.mark.parametrize("target", PRIMARIES)
@pytest.mark.parametrize("exposure", [-2.0, 0.0, 3.0])
def test_rotation_exposure_and_output_declaration(target, exposure):
    img = torch.tensor([[0.2, 0.5, 1.0], [0.0, 0.1, 0.05]], dtype=torch.float64)
    expected = _upstream_decode(img) * 2**exposure
    if target != "rec709":
        matrix = torch.tensor(MATRICES[f"rec709->{target}"], dtype=torch.float64)
        expected = expected @ matrix.T
    actual, cs = decode_ltx_hdr(img, target, exposure, clamp_negatives=False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    clipped, _ = decode_ltx_hdr(img, target, exposure)
    torch.testing.assert_close(clipped, expected.clamp_min(0), atol=1e-6, rtol=0)
    assert cs.primaries == target
    assert cs.transfer == "linear"
    assert actual.max() > 1


def test_no_upper_output_clamp_and_highlights_survive_exr(tmp_path):
    img = torch.ones((2, 3, 3), dtype=torch.float32)
    linear, cs = decode_ltx_hdr(img, exposure=2.0)
    assert linear.min().item() > 200
    path = tmp_path / "ltx.exr"
    write_exr(path, linear.numpy(), cs, half=False)
    pixels, inferred, header = read_exr(path)
    np.testing.assert_array_equal(pixels, linear.numpy())
    assert inferred.primaries == "rec709" and inferred.transfer == "linear"
    # Complete space name, not a bare transfer: this is what Nuke/OCIO key off.
    assert header["colorSpace"] == "Linear Rec.709 (sRGB)"


def test_unknown_target_is_rejected():
    with pytest.raises(KeyError, match="rec709->unknown"):
        decode_ltx_hdr(torch.ones(3), "unknown")
