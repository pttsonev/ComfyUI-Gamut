"""Linear-light primaries rotations, with Bradford white-point adaptation."""

import torch

from .colorspace import ColorSpace


# Python float literals retain float64 precision until conversion at call time.
# Rows map source RGB column vectors to destination RGB column vectors.
MATRICES = {
    "rec709->acescg": (
        (0.61309740, 0.33952315, 0.04737945),
        (0.07019372, 0.91635388, 0.01345240),
        (0.02061559, 0.10956977, 0.86981463),
    ),
    "acescg->rec709": (
        (1.70505099, -0.62179212, -0.08325887),
        (-0.13025642, 1.14080474, -0.01054832),
        (-0.02400336, -0.12896898, 1.15297233),
    ),
    "rec709->ap0": (
        (0.43963298, 0.38298870, 0.17737832),
        (0.08977644, 0.81343943, 0.09678413),
        (0.01754117, 0.11154655, 0.87091228),
    ),
    "ap0->rec709": (
        (2.52168619, -1.13413099, -0.38755520),
        (-0.27647991, 1.37271909, -0.09623917),
        (-0.01537806, -0.15297534, 1.16835340),
    ),
    "acescg->ap0": (
        (0.69545224, 0.14067870, 0.16386906),
        (0.04479456, 0.85967112, 0.09553432),
        (-0.00552588, 0.00402521, 1.00150067),
    ),
    "ap0->acescg": (
        (1.45143932, -0.23651075, -0.21492857),
        (-0.07655377, 1.17622970, -0.09967593),
        (0.00831615, -0.00603245, 0.99771630),
    ),
    "rec709->rec2020": (
        (0.62740390, 0.32928304, 0.04331307),
        (0.06909729, 0.91954040, 0.01136232),
        (0.01639144, 0.08801331, 0.89559525),
    ),
    "rec2020->rec709": (
        (1.66049100, -0.58764114, -0.07284986),
        (-0.12455047, 1.13289990, -0.00834942),
        (-0.01815076, -0.10057890, 1.11872966),
    ),
    "acescg->rec2020": (
        (1.02582475, -0.02005319, -0.00577156),
        (-0.00223437, 1.00458650, -0.00235213),
        (-0.00501335, -0.02529007, 1.03030342),
    ),
    "rec2020->acescg": (
        (0.97489498, 0.01959911, 0.00550591),
        (0.00217956, 0.99553547, 0.00228497),
        (0.00479724, 0.02453202, 0.97067074),
    ),
    "ap0->rec2020": (
        (1.49040952, -0.26617092, -0.22423860),
        (-0.08016750, 1.18216712, -0.10199962),
        (0.00322763, -0.03477648, 1.03154884),
    ),
    "rec2020->ap0": (
        (0.67908563, 0.15770091, 0.16321345),
        (0.04600200, 0.85905467, 0.09494332),
        (-0.00057394, 0.02846777, 0.97210617),
    ),
}

# EXR header order: red xy, green xy, blue xy, white xy.
CHROMATICITIES = {
    "rec709": (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290),
    "acescg": (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767),
    "ap0": (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767),
    "rec2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
}


def convert(
    img: torch.Tensor,
    src: str,
    dst: str,
    colorspace: ColorSpace | None = None,
) -> torch.Tensor:
    """Rotate linear RGB on the last axis, preserving any channels after RGB.

    Untagged input is assumed linear. A supplied ColorSpace must declare linear
    transfer, including for an identity conversion. The caller supplies the
    source and destination primaries as registry names.
    """
    if colorspace is not None and colorspace.transfer != "linear":
        raise ValueError(
            f"Cannot convert primaries with transfer {colorspace.transfer!r}; "
            "run Gamut: Transfer (decode) first to obtain linear light."
        )
    if img.ndim == 0 or img.shape[-1] < 3:
        raise ValueError("Primaries conversion requires [..., C] with C >= 3.")
    pair = f"{src}->{dst}"
    if src not in CHROMATICITIES or dst not in CHROMATICITIES:
        raise KeyError(f"Unknown primaries conversion: {pair}")
    if src == dst:
        return img
    try:
        values = MATRICES[pair]
    except KeyError:
        raise KeyError(f"Unknown primaries conversion: {pair}") from None
    matrix = torch.tensor(values, dtype=img.dtype, device=img.device)
    rgb = img[..., :3] @ matrix.T
    if img.shape[-1] == 3:
        return rgb
    return torch.cat((rgb, img[..., 3:]), dim=-1)
