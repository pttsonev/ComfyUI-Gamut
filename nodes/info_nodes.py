"""Colour declaration and EXR metadata inspection."""

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

if "." in __package__:
    from ..gamut import exr
    from ..gamut.colorspace import describe
else:
    from gamut import exr
    from gamut.colorspace import describe


class GamutColorSpaceInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": ""}),
                "colorspace": ("COLORSPACE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/Info"

    def inspect(self, image=None, path="", colorspace=None):
        lines = []
        if colorspace is not None:
            lines.append(describe(colorspace))
        if image is not None:
            lines.append(f"IMAGE shape={tuple(image.shape)}, dtype={image.dtype}, device={image.device}")
        if path:
            resolved = path
            if folder_paths is not None and not os.path.isabs(path):
                resolved = folder_paths.get_annotated_filepath(path)
            _, cs, header = exr.read_exr(resolved)
            lines.append(f"EXR: {path}")
            lines.append(describe(cs) if cs is not None else "EXR colour space: untagged or unknown")
            for key in ("chromaticities", "colorSpace"):
                if key in header:
                    lines.append(f"{key}: {header[key]!r}")
        if colorspace is None and not path:
            lines.append("No colour declaration supplied; pixel values do not identify a colour space.")
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}
