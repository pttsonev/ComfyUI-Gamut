"""Independent 16-bit image decoding and pre-quantisation dither checks."""

import numpy as np
import pytest
import torch

from gamut import highbit
from tests.conftest import DEVICES, requires_cuda


@pytest.fixture
def read16():
    # OpenCV is only an independent test decoder, never a production IO dependency.
    cv2 = pytest.importorskip("cv2", reason="Independent PNG/TIFF decoder unavailable")

    def read(path):
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        assert pixels is not None
        assert pixels.dtype == np.uint16
        order = [2, 1, 0, 3] if pixels.shape[-1] == 4 else [2, 1, 0]
        return pixels[..., order]

    return read


@pytest.mark.parametrize("writer,suffix", [(highbit.save_png16, ".png"), (highbit.save_tiff16, ".tif")])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_true_16_bit_rgb_rgba_round_trip(tmp_path, read16, writer, suffix, channels, dtype):
    # Odd low-byte values catch accidental 8-bit conversion or byte-order reversal.
    codes = torch.tensor([0, 1, 255, 256, 257, 32768, 32769, 65001, 65534, 65535])
    img = (codes.double() / 65535).reshape(2, 5, 1).expand(-1, -1, channels).to(dtype)
    img = img.clone()
    img[..., 1] = 0.25
    img[..., 2] = 0.75
    if channels == 4:
        img[..., 3] = 0.5
    original = img.clone()
    path = tmp_path / ("image" + suffix)
    writer(path, img)
    expected = (img.double() * 65535).round().numpy().astype(np.uint16)
    np.testing.assert_array_equal(read16(path), expected)
    assert torch.equal(img, original)


@pytest.mark.parametrize("writer", [highbit.save_png16, highbit.save_tiff16])
def test_noncontiguous_numpy_input_clips_instead_of_wrapping(tmp_path, read16, writer):
    img = np.array([[[-2.0, 0.0, 2.0], [0.1, 0.2, 0.3]]], np.float32)[:, ::-1]
    original = img.copy()
    path = tmp_path / "image"
    writer(path, img)
    expected = np.round(np.clip(img.astype(np.float64), 0, 1) * 65535).astype(np.uint16)
    np.testing.assert_array_equal(read16(path), expected)
    np.testing.assert_array_equal(img, original)


@pytest.mark.parametrize("writer", [highbit.save_png16, highbit.save_tiff16])
def test_dither_is_opt_in_and_applied_before_rounding(tmp_path, read16, monkeypatch, writer):
    img = torch.full((2, 3, 3), 1000 / 65535, dtype=torch.float64)
    default_path, dither_path = tmp_path / "default", tmp_path / "dither"
    rng = torch.get_rng_state().clone()
    writer(default_path, img)
    assert torch.equal(torch.get_rng_state(), rng)
    np.testing.assert_array_equal(read16(default_path), np.full((2, 3, 3), 1000))

    def deterministic_dither(values, bits):
        assert bits == 16
        assert values.is_floating_point()
        torch.testing.assert_close(values, img, atol=0, rtol=0)
        return values + 0.75 / 65535

    monkeypatch.setattr(highbit, "tpdf_dither", deterministic_dither)
    writer(dither_path, img, dither=True)
    np.testing.assert_array_equal(read16(dither_path), np.full((2, 3, 3), 1001))


@pytest.mark.parametrize("bits", [8, 10, 16])
def test_dither_has_triangular_distribution_and_one_lsb_support(bits):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(173)
        img = torch.zeros(200_000, dtype=torch.float64)
        noise = highbit.tpdf_dither(img, bits) * (2**bits - 1)
    assert noise.min() >= -1 and noise.max() <= 1
    assert abs(noise.mean().item()) < 0.003
    assert noise.var().item() == pytest.approx(1 / 6, abs=0.003)
    assert (noise.abs() < 0.5).double().mean().item() == pytest.approx(0.75, abs=0.005)
    assert torch.count_nonzero(img) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dither_preserves_shape_dtype_device_and_does_not_clip(device, dtype):
    img = torch.tensor([-1.0, 0.0, 1.0, 100.0], dtype=dtype, device=device)
    original = img.clone()
    actual = highbit.tpdf_dither(img, 16)
    assert (actual.shape, actual.dtype, actual.device) == (img.shape, img.dtype, img.device)
    assert actual[0] < 0 and actual[-1] > 1
    assert torch.equal(img, original)


@requires_cuda
def test_gpu_input_can_be_saved_without_mutation(tmp_path, read16):
    img = torch.full((2, 3, 3), 0.5, device="cuda")
    original = img.clone()
    path = tmp_path / "gpu.png"
    highbit.save_png16(path, img)
    np.testing.assert_array_equal(read16(path), np.full((2, 3, 3), 32768))
    assert torch.equal(img, original)


@pytest.mark.parametrize("bits", [0, -1, 2.5, True])
def test_invalid_dither_bit_depth_is_rejected(bits):
    with pytest.raises(ValueError, match="positive integer"):
        highbit.tpdf_dither(torch.zeros(3), bits)


@pytest.mark.parametrize("writer", [highbit.save_png16, highbit.save_tiff16])
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 2), (0, 3, 3), (1, 2, 3, 3)])
def test_invalid_image_shape_is_rejected(tmp_path, writer, shape):
    with pytest.raises(ValueError, match="nonempty"):
        writer(tmp_path / "invalid", torch.zeros(shape))


@pytest.mark.parametrize("writer", [highbit.save_png16, highbit.save_tiff16])
def test_integer_nonfinite_and_unwritable_inputs_raise(tmp_path, writer):
    with pytest.raises(TypeError, match="floating-point"):
        writer(tmp_path / "integer", torch.zeros((2, 3, 3), dtype=torch.int32))
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            writer(tmp_path / "nonfinite", torch.full((2, 3, 3), value))
    with pytest.raises(OSError):
        writer(tmp_path / "missing" / "image", torch.zeros((2, 3, 3)))
