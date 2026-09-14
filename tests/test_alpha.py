"""Opacity is independent of colour: polarity, extraction and storage ownership."""

import pytest
import torch

from gamut.alpha import embed_alpha, extract_alpha


@pytest.mark.parametrize("channels", [3, 4])
def test_mask_polarity_and_replacement_preserve_colour(channels):
    image = torch.linspace(-2, 7, 2 * channels).reshape(1, 1, 2, channels)
    original = image.clone()
    mask = torch.tensor([[[0., 1.]]])
    opacity = extract_alpha(image, alpha_mask=mask)
    rgba = embed_alpha(image, opacity)
    torch.testing.assert_close(rgba[..., 3], torch.tensor([[[1., 0.]]]))
    torch.testing.assert_close(rgba[..., :3], original[..., :3], atol=0, rtol=0)
    rgba.zero_()
    torch.testing.assert_close(image, original, atol=0, rtol=0)
    torch.testing.assert_close(mask, torch.tensor([[[0., 1.]]]))


@pytest.mark.parametrize("component,index", [("R", 0), ("G", 1), ("B", 2), ("A", 3)])
def test_image_opacity_uses_selected_component_without_inversion(component, index):
    image = torch.ones(1, 1, 1, 3)
    matte = torch.tensor([[[[0.1, 0.3, 0.6, 0.9]]]])
    opacity = extract_alpha(image, alpha_image=matte, alpha_channel=component)
    rgba = embed_alpha(image, opacity)
    torch.testing.assert_close(rgba[..., 3], matte[..., index])
    rgba.zero_()
    torch.testing.assert_close(matte, torch.tensor([[[[0.1, 0.3, 0.6, 0.9]]]]))


def test_scalar_opacity_and_absent_alpha():
    image = torch.tensor([[[[2., -1., 4., 9.]]]])
    assert extract_alpha(image) is None
    assert embed_alpha(image) is image  # legacy A is neither clamped nor copied
    matte = torch.tensor([[[[0.25]]]])
    torch.testing.assert_close(embed_alpha(image, extract_alpha(image, alpha_image=matte)),
                               torch.tensor([[[[2., -1., 4., 0.25]]]]))


def test_alpha_precision_is_not_reduced_to_the_input_rgb_dtype():
    image = torch.ones(1, 1, 1, 3, dtype=torch.float16)
    opacity = torch.tensor([[[0.1234567]]], dtype=torch.float32)
    result = embed_alpha(image, opacity)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result[..., 3], opacity, atol=0, rtol=0)


@pytest.mark.parametrize("kwargs,match", [
    ({"alpha_mask": torch.zeros(1, 1, 1), "alpha_image": torch.zeros(1, 1, 1, 3)}, "at most one"),
    ({"alpha_mask": torch.zeros(1, 2, 1)}, "alpha_mask expected"),
    ({"alpha_mask": torch.zeros(1, 1)}, "alpha_mask expected"),
    ({"alpha_mask": torch.zeros(1, 1, 1, dtype=torch.int32)}, "floating-point"),
    ({"alpha_image": torch.zeros(2, 1, 1, 3)}, "alpha_image expected"),
    ({"alpha_image": torch.zeros(1, 1, 1, 2)}, "alpha_image requires"),
    ({"alpha_image": torch.zeros(1, 1, 1, 3), "alpha_channel": "A"}, "no selected A"),
    ({"alpha_image": torch.zeros(1, 1, 1, 1), "alpha_channel": "G"}, "no selected G"),
    ({"alpha_image": torch.zeros(1, 1, 1, 3), "alpha_channel": "luma"}, "Unknown alpha_channel"),
])
def test_invalid_alpha_is_refused(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        extract_alpha(torch.zeros(1, 1, 1, 3), **kwargs)


@pytest.mark.parametrize("value", [-0.001, 1.001, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("source", ["alpha_mask", "alpha_image"])
def test_external_alpha_must_be_finite_and_bounded(value, source):
    shape = (1, 1, 1) if source == "alpha_mask" else (1, 1, 1, 1)
    with pytest.raises(ValueError, match="finite opacity"):
        extract_alpha(torch.ones(1, 1, 1, 3), **{source: torch.full(shape, value)})
