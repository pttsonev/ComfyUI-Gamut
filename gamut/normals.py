"""Pure mappings between IMAGE codes and signed normal vectors."""

import torch


def _orient_normals(
    vectors: torch.Tensor,
    renormalise: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> torch.Tensor:
    if vectors.ndim == 0 or vectors.shape[-1] != 3:
        raise ValueError("Normal conversion requires [..., 3] vectors.")
    if renormalise:
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        vectors = vectors / torch.where(lengths == 0, torch.ones_like(lengths), lengths)
    if flip_x or flip_y or flip_z:
        signs = vectors.new_tensor((
            -1.0 if flip_x else 1.0,
            -1.0 if flip_y else 1.0,
            -1.0 if flip_z else 1.0,
        ))
        vectors = vectors * signs
    return vectors


def decode_normals(
    image: torch.Tensor,
    renormalise: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> torch.Tensor:
    """Map [..., 3] IMAGE codes in [0, 1] to signed vectors, including batches.

    Renormalisation and axis flips are opt-in; no model convention is assumed.
    Zero vectors stay zero when renormalising. Float32 and device are preserved,
    and the input is never modified or clipped.
    """
    vectors = image * 2.0 - 1.0
    return _orient_normals(vectors, renormalise, flip_x, flip_y, flip_z)


def encode_normals(
    vectors: torch.Tensor,
    renormalise: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> torch.Tensor:
    """Map [..., 3] signed vectors to IMAGE codes for display, including batches.

    Matching axis flips undo decode_normals; renormalisation discards length.
    Zero vectors encode as 0.5. Float32 and device are preserved, and the input
    is never modified or clipped.
    """
    vectors = _orient_normals(vectors, renormalise, flip_x, flip_y, flip_z)
    return (vectors + 1.0) * 0.5
