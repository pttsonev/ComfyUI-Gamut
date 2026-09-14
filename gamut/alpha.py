"""Opacity assembly for colour writers; no colour or premultiplication operations."""

import torch


def validate_image(image, name="image", channels=(3, 4)):
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise TypeError(f"{name} requires a floating-point IMAGE tensor.")
    if image.ndim != 4 or image.shape[-1] not in channels or 0 in image.shape:
        raise ValueError(
            f"{name} requires a nonempty IMAGE [B, H, W, {channels}]; got {tuple(image.shape)}."
        )


def extract_alpha(image, alpha_mask=None, alpha_image=None, alpha_channel="R"):
    """Validate a batch and return external opacity, or None to preserve existing A.

    No copy of the colour batch is made. The caller can embed opacity a frame at
    a time, even when the matte and image arrive on different devices.
    """
    validate_image(image)
    if alpha_mask is not None and alpha_image is not None:
        raise ValueError("Connect at most one of alpha_mask and alpha_image.")
    if alpha_mask is None and alpha_image is None:
        return None
    if alpha_mask is not None:
        if not isinstance(alpha_mask, torch.Tensor) or not alpha_mask.is_floating_point():
            raise TypeError("alpha_mask requires a floating-point MASK tensor.")
        if tuple(alpha_mask.shape) != tuple(image.shape[:3]):
            raise ValueError(f"alpha_mask expected {tuple(image.shape[:3])}; got {tuple(alpha_mask.shape)}.")
        opacity = 1.0 - alpha_mask
    else:
        validate_image(alpha_image, "alpha_image", (1, 3, 4))
        if tuple(alpha_image.shape[:3]) != tuple(image.shape[:3]):
            raise ValueError(f"alpha_image expected {tuple(image.shape[:3])}; got {tuple(alpha_image.shape[:3])}.")
        if alpha_channel not in ("R", "G", "B", "A"):
            raise ValueError(f"Unknown alpha_channel: {alpha_channel!r}.")
        index = "RGBA".index(alpha_channel)
        if index >= alpha_image.shape[-1]:
            raise ValueError(f"alpha_image has no selected {alpha_channel} component.")
        opacity = alpha_image[..., index]
    if not bool(torch.isfinite(opacity).all()) or bool(((opacity < 0) | (opacity > 1)).any()):
        raise ValueError("External alpha must contain finite opacity values in [0,1].")
    return opacity


def embed_alpha(image, opacity=None):
    """Embed one frame or batch without modifying caller storage or multiplying RGB."""
    if opacity is None:
        return image
    # Preserve matte precision until the writer applies the selected output dtype.
    return torch.cat((image[..., :3], opacity.to(device=image.device).unsqueeze(-1)), dim=-1)
