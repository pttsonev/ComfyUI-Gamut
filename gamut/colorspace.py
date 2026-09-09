"""Colour declarations carried beside image tensors, including their provenance."""

from dataclasses import dataclass


PRIMARIES = {
    "rec709": "Rec.709",
    "acescg": "ACEScg",
    "ap0": "ACES2065-1 (AP0)",
    "rec2020": "Rec.2020",
}

TRANSFERS = {
    "linear": "Linear",
    "srgb": "sRGB",
    "logc3": "ARRI LogC3 (EI800)",
    "acescct": "ACEScct",
    "g22": "Gamma 2.2",
    "g24": "Gamma 2.4",
}


@dataclass(frozen=True)
class ColorSpace:
    """Declared primaries, transfer curve, and how that declaration is known."""

    primaries: str
    transfer: str
    source: str


def describe(cs: ColorSpace) -> str:
    """Describe the colour state without losing its provenance or unknown names."""
    primaries = PRIMARIES.get(cs.primaries, cs.primaries)
    transfer = TRANSFERS.get(cs.transfer, cs.transfer)
    return f"Primaries: {primaries}; transfer: {transfer}; source: {cs.source}"
