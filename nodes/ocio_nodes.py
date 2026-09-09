"""Optional OCIO nodes. Importing the definitions does not require OCIO."""

from functools import lru_cache

if "." in __package__:
    from ..gamut import ocio_bridge
    from ..gamut.exr import _CANONICAL_NAMES
else:
    from gamut import ocio_bridge
    from gamut.exr import _CANONICAL_NAMES


@lru_cache(maxsize=1)
def _choices():
    if not ocio_bridge.available():
        return list(_CANONICAL_NAMES.values()), [""], [""]
    cfg = ocio_bridge.get_config()
    spaces = ocio_bridge.list_spaces(cfg)
    displays = list(cfg.getDisplays())
    views = list(dict.fromkeys(view for display in displays for view in cfg.getViews(display)))
    return spaces, displays or [""], views or [""]


def _validate_connections(input_types):
    for name, received in (input_types or {}).items():
        if name == "image":
            allowed = ("IMAGE",)
        elif name == "colorspace":
            allowed = ("COLORSPACE",)
        elif name == "config":
            allowed = ("STRING",)
        else:
            allowed = ("STRING", "COMBO")
        if received not in allowed:
            return f"{name} requires {' or '.join(allowed)}, received {received}."
    return True


class GamutOCIOTransform:
    @classmethod
    def INPUT_TYPES(cls):
        spaces, _, _ = _choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "config": ("STRING", {"default": ocio_bridge.DEFAULT_CONFIG}),
                "src": (spaces, {"default": _CANONICAL_NAMES[("rec709", "linear")]}),
                "dst": (spaces, {"default": _CANONICAL_NAMES[("acescg", "linear")]}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE", "COLORSPACE")
    RETURN_NAMES = ("image", "colorspace")
    FUNCTION = "transform"
    CATEGORY = "Gamut/OCIO"

    @classmethod
    def VALIDATE_INPUTS(cls, src=None, dst=None, input_types=None):
        # Custom configs may have names absent from the default config dropdowns.
        # OCIO validates the actual source/destination against the selected config.
        return _validate_connections(input_types)

    def transform(self, image, config, src, dst, colorspace=None):
        cfg = ocio_bridge.get_config(config)
        source = ocio_bridge.source_name(colorspace, cfg) if colorspace is not None else src
        result = ocio_bridge.apply_transform(image.detach().cpu().numpy(), cfg, source, dst)
        return ocio_bridge.image_from_array(result, image), ocio_bridge.declaration(cfg, dst)


class GamutOCIODisplayView:
    @classmethod
    def INPUT_TYPES(cls):
        spaces, displays, views = _choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "config": ("STRING", {"default": ocio_bridge.DEFAULT_CONFIG}),
                "src": (spaces, {
                    "default": _CANONICAL_NAMES[("rec709", "linear")],
                    "tooltip": "Explicit source space. A wired COLORSPACE takes precedence.",
                }),
                "display": (displays,),
                "view": (views,),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "display"
    CATEGORY = "Gamut/OCIO"

    @classmethod
    def VALIDATE_INPUTS(cls, src=None, display=None, view=None, input_types=None):
        return _validate_connections(input_types)

    def display(self, image, config, src, display, view, colorspace=None):
        cfg = ocio_bridge.get_config(config)
        source = ocio_bridge.source_name(colorspace, cfg) if colorspace is not None else src
        result = ocio_bridge.apply_display(image.detach().cpu().numpy(), cfg, source, display, view)
        return (ocio_bridge.image_from_array(result, image),)


class GamutApplyLUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"default": ""}),
                "direction": (["forward", "inverse"], {"default": "forward"}),
                "interpolation": (["linear", "nearest", "tetrahedral", "best"], {"default": "linear"}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Gamut/OCIO"

    def apply(self, image, lut_path, direction="forward", interpolation="linear", colorspace=None):
        result = ocio_bridge.apply_file_transform(
            image.detach().cpu().numpy(), lut_path, direction, interpolation
        )
        return (ocio_bridge.image_from_array(result, image),)
