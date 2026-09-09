"""ComfyUI-Gamut node registration, with optional OCIO capability detection."""

import logging

if __package__:
    from .gamut import ColorSpace, ocio_bridge
    from .nodes.core_nodes import GamutColorSpaceConvert, GamutTransfer
    from .nodes.info_nodes import GamutColorSpaceInfo
    from .nodes.io_nodes import GamutLoadEXR, GamutSaveEXR, GamutSaveHighBit
    from .nodes.ltx_nodes import GamutLTXHDRDecode
else:
    # pytest may load a checkout with a hyphenated directory name as __init__.
    from gamut import ColorSpace, ocio_bridge
    from nodes.core_nodes import GamutColorSpaceConvert, GamutTransfer
    from nodes.info_nodes import GamutColorSpaceInfo
    from nodes.io_nodes import GamutLoadEXR, GamutSaveEXR, GamutSaveHighBit
    from nodes.ltx_nodes import GamutLTXHDRDecode


# Classic ComfyUI custom types are registered by matching socket type strings.
# This map also exposes the Python payload type to consumers of the package.
CUSTOM_TYPES = {"COLORSPACE": ColorSpace}

NODE_CLASS_MAPPINGS = {
    "GamutColorSpaceConvert": GamutColorSpaceConvert,
    "GamutTransfer": GamutTransfer,
    "GamutLoadEXR": GamutLoadEXR,
    "GamutSaveEXR": GamutSaveEXR,
    "GamutSaveHighBit": GamutSaveHighBit,
    "GamutLTXHDRDecode": GamutLTXHDRDecode,
    "GamutColorSpaceInfo": GamutColorSpaceInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GamutColorSpaceConvert": "Gamut: Color Space Convert",
    "GamutTransfer": "Gamut: Transfer",
    "GamutLoadEXR": "Gamut: Load EXR",
    "GamutSaveEXR": "Gamut: Save EXR",
    "GamutSaveHighBit": "Gamut: Save High Bit",
    "GamutLTXHDRDecode": "Gamut: LTX HDR Decode",
    "GamutColorSpaceInfo": "Gamut: Color Space Info",
}

if ocio_bridge.available():
    if __package__:
        from .nodes.ocio_nodes import GamutApplyLUT, GamutOCIODisplayView, GamutOCIOTransform
    else:
        from nodes.ocio_nodes import GamutApplyLUT, GamutOCIODisplayView, GamutOCIOTransform
    NODE_CLASS_MAPPINGS.update({
        "GamutOCIOTransform": GamutOCIOTransform,
        "GamutOCIODisplayView": GamutOCIODisplayView,
        "GamutApplyLUT": GamutApplyLUT,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "GamutOCIOTransform": "Gamut: OCIO Transform",
        "GamutOCIODisplayView": "Gamut: OCIO Display View",
        "GamutApplyLUT": "Gamut: Apply LUT",
    })
elif not globals().get("_OCIO_NOTICE_EMITTED", False):
    logging.getLogger(__name__).info(
        "Gamut: OCIO nodes disabled because the optional opencolorio binding is unavailable."
    )
    _OCIO_NOTICE_EMITTED = True

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "CUSTOM_TYPES"]
