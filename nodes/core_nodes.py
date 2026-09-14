"""Colour declaration nodes and normal-vector encode/decode adapters."""

# Support both ComfyUI's package loader and standalone imports in development.
if "." in __package__:
    from ..gamut import normals, primaries, transfer
    from ..gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
else:
    from gamut import normals, primaries, transfer
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


class GamutNormalDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Normal codes in [0, 1]; output is signed vector data."}),
                "renormalise": ("BOOLEAN", {"default": False}),
                "flip_x": ("BOOLEAN", {"default": False}),
                "flip_y": ("BOOLEAN", {"default": False}),
                "flip_z": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normals",)
    FUNCTION = "decode"
    CATEGORY = "Gamut/Data"
    DESCRIPTION = "Map [0, 1] normal codes to signed vectors carried as an IMAGE."

    def decode(self, image, renormalise=False, flip_x=False, flip_y=False, flip_z=False):
        return (normals.decode_normals(
            image, renormalise=renormalise, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z,
        ),)


class GamutNormalEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Signed normal vectors carried as an IMAGE."}),
                "renormalise": ("BOOLEAN", {"default": False}),
                "flip_x": ("BOOLEAN", {"default": False}),
                "flip_y": ("BOOLEAN", {"default": False}),
                "flip_z": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "encode"
    CATEGORY = "Gamut/Data"
    DESCRIPTION = "Map signed normal vectors to IMAGE codes for display."

    def encode(self, image, renormalise=False, flip_x=False, flip_y=False, flip_z=False):
        return (normals.encode_normals(
            image, renormalise=renormalise, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z,
        ),)
