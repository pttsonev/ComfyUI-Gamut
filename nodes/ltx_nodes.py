"""LTX HDR decoding adapter."""

if "." in __package__:
    from ..gamut.colorspace import PRIMARIES
    from ..gamut.ltx import decode_ltx_hdr
else:
    from gamut.colorspace import PRIMARIES
    from gamut.ltx import decode_ltx_hdr


class GamutLTXHDRDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "LogC3 HDR IC-LoRA VAE output in [0, 1]."}),
                "target_primaries": (list(PRIMARIES), {"default": "rec709"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE", "COLORSPACE")
    RETURN_NAMES = ("image", "colorspace")
    FUNCTION = "decode"
    CATEGORY = "Gamut/LTX"

    def decode(self, image, target_primaries="rec709", exposure=0.0, colorspace=None):
        if colorspace is not None and (
            colorspace.primaries != "rec709" or colorspace.transfer != "logc3"
        ):
            raise ValueError("LTX HDR decoding requires Rec.709/LogC3 VAE codes.")
        return decode_ltx_hdr(image, target_primaries, exposure)
