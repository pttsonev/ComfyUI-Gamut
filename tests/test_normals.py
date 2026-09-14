"""Signed-normal mappings: exact round trips, orientation, and what is not assumed.

The defaults matter as much as the maths here. UniVidX's normal convention —
coordinate frame and handedness — is unmeasured at the time of writing, so
``decode_normals`` must apply no correction unless asked. A test suite that
only exercised the flipped paths would let a "helpful" default slip in.
"""

import itertools

import pytest
import torch

from gamut.normals import decode_normals, encode_normals


FLIPS = list(itertools.product([False, True], repeat=3))


def _codes(shape=(2, 4, 5, 3), device="cpu", dtype=torch.float32):
    """Deterministic codes spanning [0, 1] inclusive, so both ends are covered."""
    n = 1
    for dim in shape:
        n *= dim
    return torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(shape)


@pytest.mark.parametrize("flip_x,flip_y,flip_z", FLIPS)
def test_flip_only_round_trip_recovers_input(flip_x, flip_y, flip_z, device):
    """encode(decode(x)) recovers x for every flip combination, to float tolerance.

    Not bit-exact, and deliberately not asserted as such: ``*2-1`` followed by
    ``+1`` loses low bits for codes near zero, so a float32 ``linspace``
    starting at 0.0 round-trips to ~1.5e-08 rather than exactly. (Uniform
    random input happens to be bit-exact, which is what makes this easy to
    over-assert.) The tolerance is still four orders tighter than any clamp,
    sign error, or stray renormalisation would produce.
    """
    codes = _codes(device=device)
    kw = dict(flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
    out = encode_normals(decode_normals(codes, **kw), **kw)
    assert torch.allclose(out, codes, rtol=0, atol=1e-7)


def test_default_decode_applies_no_orientation(device):
    """With no flags, decode is exactly the range map and nothing else."""
    codes = _codes(device=device)
    assert torch.equal(decode_normals(codes), codes * 2.0 - 1.0)


@pytest.mark.parametrize("axis,flag", [(0, "flip_x"), (1, "flip_y"), (2, "flip_z")])
def test_each_flip_negates_only_its_own_axis(axis, flag, device):
    """A flip must be independent — three separate axes, not one shared sign."""
    codes = _codes(device=device)
    plain = decode_normals(codes)
    flipped = decode_normals(codes, **{flag: True})
    for other in range(3):
        expected = -plain[..., other] if other == axis else plain[..., other]
        assert torch.equal(flipped[..., other], expected)


def test_renormalise_produces_unit_length(device):
    """Renormalising yields unit vectors; the default path does not."""
    codes = _codes(device=device)
    unit = decode_normals(codes, renormalise=True)
    lengths = unit.norm(dim=-1)
    # Every row of _codes is non-zero somewhere, so all should reach unit length.
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-6)
    plain_lengths = decode_normals(codes).norm(dim=-1)
    assert not torch.allclose(plain_lengths, torch.ones_like(plain_lengths), atol=1e-3)


def test_renormalise_discards_length_so_round_trip_is_not_identity(device):
    """Documented lossy case: renormalising throws magnitude away.

    This is the one round trip that must NOT be an identity. Pinning it stops
    someone "fixing" the asymmetry by carrying length through, which would
    silently make renormalise a no-op.
    """
    codes = _codes(device=device)
    out = encode_normals(decode_normals(codes, renormalise=True), renormalise=True)
    assert not torch.allclose(out, codes, atol=1e-3)


def test_zero_vectors_survive_renormalisation(device):
    """A 0.5 code decodes to the zero vector; dividing by its length must not NaN."""
    codes = torch.full((2, 3, 3), 0.5, device=device)
    out = decode_normals(codes, renormalise=True)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_zero_vector_encodes_to_mid_grey(device):
    """The inverse of the above: a zero vector is the 0.5 code."""
    vectors = torch.zeros(2, 3, 3, device=device)
    assert torch.equal(encode_normals(vectors), torch.full_like(vectors, 0.5))


def test_values_are_never_clipped(device):
    """Out-of-range input passes through untouched, both directions.

    Clamping here would hide a miscalibrated producer instead of surfacing it.
    """
    codes = torch.tensor([[[2.0, -1.0, 0.5]]], device=device)
    assert torch.equal(decode_normals(codes), torch.tensor([[[3.0, -3.0, 0.0]]], device=device))
    vectors = torch.tensor([[[4.0, -4.0, 0.0]]], device=device)
    assert torch.equal(encode_normals(vectors), torch.tensor([[[2.5, -1.5, 0.5]]], device=device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fn", [decode_normals, encode_normals])
def test_preserves_shape_dtype_device_and_input(fn, dtype, device):
    """Neither direction may mutate its argument or change dtype/device/shape."""
    src = _codes(device=device, dtype=dtype)
    before = src.clone()
    out = fn(src, renormalise=True, flip_y=True)
    assert out.shape == src.shape
    assert out.dtype == dtype
    assert out.device == src.device
    assert torch.equal(src, before)


@pytest.mark.parametrize("shape", [(3,), (4, 3), (2, 4, 5, 3), (1, 2, 3, 4, 3)])
def test_accepts_any_trailing_three_shape(shape, device):
    """Batch-capable by contract: anything shaped [..., 3], bare vector included."""
    codes = _codes(shape=shape, device=device)
    assert decode_normals(codes).shape == codes.shape


@pytest.mark.parametrize("shape", [(2, 4, 5, 4), (2, 4, 5), (2, 4, 5, 1), ()])
@pytest.mark.parametrize("fn", [decode_normals, encode_normals])
def test_rejects_shapes_that_are_not_three_vectors(fn, shape, device):
    """A wrong trailing dimension is a caller error, never a broadcast."""
    bad = torch.zeros(shape, device=device)
    with pytest.raises(ValueError, match=r"\[\.\.\., 3\]"):
        fn(bad)


def test_module_is_importable_without_comfyui():
    """The helpers stay pure so they remain testable outside ComfyUI.

    Importing gamut.normals must not drag in folder_paths or any other
    ComfyUI-only module.
    """
    import sys

    import gamut.normals  # noqa: F401  (already imported; this pins the contract)

    assert "folder_paths" not in sys.modules
