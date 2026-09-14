"""Transfer reference fixtures, extended-domain round trips, and regressions."""

import pytest
import torch

from gamut.colorspace import TRANSFERS
from gamut.transfer import (
    acescct_decode,
    acescct_encode,
    decode,
    encode,
    gamma_decode,
    gamma_encode,
    logc3_decode,
    logc3_encode,
    srgb_decode,
    srgb_encode,
)


from .conftest import DEVICES  # probed, not just is_available()


def _lightricks_logc3_encode(linear):
    # Independent upstream EI800 formula: do not import production constants.
    a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
    e, f, cut = 5.367655, 0.092809, 0.010591
    return torch.where(
        linear > cut, c * torch.log10(a * linear + b) + d, e * linear + f
    )


def _lightricks_logc3_decode(codes):
    a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
    e, f, cut = 5.367655, 0.092809, 0.010591
    return torch.where(
        codes > e * cut + f,
        (torch.pow(10.0, (codes - d) / c) - b) / a,
        (codes - f) / e,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_logc3_matches_exact_lightricks_constants_on_21_step_ramps(dtype, device):
    ramp = torch.linspace(0, 1, 21, dtype=dtype, device=device)
    # Exact equality is stronger than the plan's 1e-6 reference tolerance.
    assert torch.equal(logc3_encode(ramp), _lightricks_logc3_encode(ramp))
    assert torch.equal(logc3_decode(ramp), _lightricks_logc3_decode(ramp))


@pytest.mark.parametrize("curve", TRANSFERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_curve_round_trips_linear_and_encoded_values(curve, dtype):
    values = torch.tensor(
        [-0.5, -0.01, 0.0, 0.001, 0.0031308, 0.0078125, 0.010591,
         0.04, 0.18, 0.5, 1.0, 2.0], dtype=dtype,
    )
    torch.testing.assert_close(decode(encode(values, curve), curve), values, atol=1e-6, rtol=0)
    codes = torch.linspace(-0.2, 1.2, 21, dtype=dtype)
    torch.testing.assert_close(encode(decode(codes, curve), curve), codes, atol=1e-6, rtol=0)


def test_srgb_known_pairs_and_piecewise_thresholds():
    linear = torch.tensor(
        [-0.1, 0.0, 0.001, 0.0031308, 0.18, 0.21404114048223255, 1.0],
        dtype=torch.float64,
    )
    codes = torch.tensor(
        [-1.292, 0.0, 0.01292, 0.040449936, 0.46135612950044164, 0.5, 1.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(srgb_encode(linear), codes, atol=1e-6, rtol=0)
    torch.testing.assert_close(srgb_decode(codes), linear, atol=1e-6, rtol=0)
    threshold = torch.tensor(0.04044823627710817, dtype=torch.float64)
    assert srgb_decode(threshold).item() == (threshold / 12.92).item()
    above = threshold + 1e-8
    assert srgb_decode(above).item() == (((above + 0.055) / 1.055) ** 2.4).item()


def test_logc3_decode_half_is_0513383_without_vae_rescale_regression():
    code = torch.tensor(0.5, dtype=torch.float64)
    assert logc3_decode(code).item() == pytest.approx(0.513383, abs=1e-6)
    assert logc3_decode(code * 2 - 1).item() == pytest.approx(-0.017290, abs=1e-6)


def test_logc3_decode_is_unclamped_below_zero_and_above_one_regression():
    codes = torch.tensor([-0.5, -0.1, 0.0, 1.0, 1.0640162984, 1.2], dtype=torch.float64)
    linear = logc3_decode(codes)
    torch.testing.assert_close(linear, _lightricks_logc3_decode(codes), atol=1e-6, rtol=0)
    assert linear[0] < linear[2] < 0
    assert linear[-1] > linear[3] > 1
    torch.testing.assert_close(logc3_encode(linear), codes, atol=1e-6, rtol=0)


def test_logc3_linear_100_highlight_survives_codes_above_one_regression():
    linear = torch.tensor(100.0, dtype=torch.float64)
    code = logc3_encode(linear)
    assert code.item() == pytest.approx(1.064016298400436, abs=1e-6)
    torch.testing.assert_close(logc3_decode(code), linear, atol=1e-6, rtol=0)
    assert logc3_decode(torch.tensor(1.0, dtype=torch.float64)).item() == pytest.approx(
        55.079577, abs=1e-6
    )


def test_logc3_piecewise_cut_and_adjacent_values():
    linear = torch.tensor([0.010590, 0.010591, 0.010592], dtype=torch.float64)
    codes = torch.tensor([0.149656834, 0.149657834105, 0.149658834], dtype=torch.float64)
    torch.testing.assert_close(
        logc3_encode(linear), _lightricks_logc3_encode(linear), atol=0, rtol=0
    )
    torch.testing.assert_close(
        logc3_decode(codes), _lightricks_logc3_decode(codes), atol=0, rtol=0
    )


def test_acescct_known_pairs_and_breakpoints():
    linear = torch.tensor([-0.1, 0.0, 0.0078125, 0.18, 1.0, 16.0], dtype=torch.float64)
    codes = torch.tensor(
        [-0.9811182399696145, 0.0729055341958355, 0.155251141552511,
         0.4135884024924423, 0.5547945205479452, 0.7831050228310502],
        dtype=torch.float64,
    )
    torch.testing.assert_close(acescct_encode(linear), codes, atol=1e-6, rtol=0)
    torch.testing.assert_close(acescct_decode(codes), linear, atol=1e-6, rtol=0)
    near_break = torch.tensor([0.0078124, 0.0078125, 0.0078126], dtype=torch.float64)
    torch.testing.assert_close(
        acescct_decode(acescct_encode(near_break)), near_break, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("g", [2.2, 2.4])
def test_gamma_is_sign_preserving_in_both_directions(g):
    values = torch.tensor([-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0], dtype=torch.float64)
    encoded = gamma_encode(values, g)
    decoded = gamma_decode(values, g)
    assert torch.isfinite(encoded).all()
    assert torch.isfinite(decoded).all()
    assert torch.equal(encoded.sign(), values.sign())
    assert torch.equal(decoded.sign(), values.sign())
    assert encoded[2].item() == pytest.approx(-(0.25 ** (1.0 / g)))
    assert decoded[2].item() == pytest.approx(-(0.25 ** g))
    torch.testing.assert_close(gamma_decode(encoded, g), values, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "forward,inverse",
    [
        (srgb_encode, srgb_decode),
        (logc3_encode, logc3_decode),
        (acescct_encode, acescct_decode),
    ],
)
def test_piecewise_curves_preserve_negative_inputs_with_finite_gradients(forward, inverse):
    linear = torch.tensor([-2.0, -0.1, -0.01, 0.0], dtype=torch.float64, requires_grad=True)
    encoded = forward(linear)
    restored = inverse(encoded)
    assert torch.isfinite(encoded).all()
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, linear, atol=1e-6, rtol=0)
    restored.sum().backward()
    assert torch.isfinite(linear.grad).all()
    negative_codes = torch.tensor([-2.0, -0.1, -0.01], dtype=torch.float64)
    decoded = inverse(negative_codes)
    assert torch.isfinite(decoded).all()
    assert (decoded < 0).all()


@pytest.mark.parametrize("curve", TRANSFERS)
def test_dispatch_preserves_negatives_by_default_and_clamps_only_when_requested(curve):
    linear = torch.tensor([-1.0, -0.01, 0.0, 0.18, 2.0], dtype=torch.float64)
    original = linear.clone()
    codes = encode(linear, curve)
    torch.testing.assert_close(decode(codes, curve), linear, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        encode(linear, curve, clamp_negatives=True), encode(linear.clamp_min(0), curve),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        decode(codes, curve, clamp_negatives=True), linear.clamp_min(0), atol=1e-6, rtol=0
    )
    assert torch.equal(linear, original)


@pytest.mark.parametrize("curve", TRANSFERS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_transfers_preserve_shape_dtype_device_and_input(curve, dtype, device):
    img = torch.linspace(-0.5, 2.0, 72, dtype=dtype, device=device).reshape(2, 3, 4, 3)
    img = img.transpose(1, 2)
    original = img.clone()
    for result in (encode(img, curve), decode(img, curve)):
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert result.device == img.device
        assert torch.isfinite(result).all()
    assert torch.equal(img, original)


def test_linear_transfer_is_identity():
    img = torch.tensor([-0.5, 0.0, 100.0])
    assert encode(img, "linear") is img
    assert decode(img, "linear") is img


@pytest.mark.parametrize("transform", [encode, decode])
def test_unknown_curve_error_names_the_curve(transform):
    with pytest.raises(KeyError, match="unknown-curve"):
        transform(torch.ones(3), "unknown-curve")
