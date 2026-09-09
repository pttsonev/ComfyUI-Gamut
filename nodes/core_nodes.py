"""Primaries and transfer nodes with explicit colour declarations."""

# Support both ComfyUI's package loader and standalone imports in development.
if "." in __package__:
    from ..gamut import primaries, transfer
    from ..gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
else:
    from gamut import primaries, transfer
    from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace


class GamutColorSpaceConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Untagged input is assumed linear light."}),
                "from_primaries": (list(PRIMARIES), {"default": "rec709"}),
                "to_primaries": (list(PRIMARIES), {"default": "acescg"}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE", "COLORSPACE")
    RETURN_NAMES = ("image", "colorspace")
    FUNCTION = "convert"
    CATEGORY = "Gamut/Colour"
    DESCRIPTION = "Rotate linear-light primaries. Untagged input is assumed linear."

    def convert(self, image, from_primaries="rec709", to_primaries="acescg", colorspace=None):
        cs = colorspace or ColorSpace(from_primaries, "linear", "widget")
        result = primaries.convert(image, cs.primaries, to_primaries, colorspace=cs)
        return result, ColorSpace(to_primaries, "linear", "node:GamutColorSpaceConvert")


class GamutTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["decode", "encode"], {"default": "decode"}),
                "curve": (list(TRANSFERS), {"default": "srgb"}),
                "primaries": (list(PRIMARIES), {"default": "rec709"}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE", "COLORSPACE")
    RETURN_NAMES = ("image", "colorspace")
    FUNCTION = "apply"
    CATEGORY = "Gamut/Colour"
    DESCRIPTION = "Decode to linear light or encode linear light; preserve negatives and alpha."

    def apply(self, image, direction="decode", curve="srgb", primaries="rec709", colorspace=None):
        if direction not in ("encode", "decode"):
            raise ValueError(f"Unknown transfer direction: {direction!r}")
        cs = colorspace or ColorSpace(
            primaries, curve if direction == "decode" else "linear", "widget"
        )
        if direction == "encode" and cs.transfer != "linear":
            raise ValueError(f"Input transfer is {cs.transfer!r}; run Gamut: Transfer (decode) first.")
        selected = cs.transfer if direction == "decode" else curve
        result = transfer.transform_rgb(image, selected, direction)
        output_transfer = "linear" if direction == "decode" else curve
        return result, ColorSpace(cs.primaries, output_transfer, "node:GamutTransfer")
