"""EXR and high-bit file nodes; ComfyUI path handling stays at this boundary."""

import os
from functools import partial
from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None

if "." in __package__:
    from ..gamut import exr, highbit
    from ..gamut import transfer as transfer_curves
    from ..gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
else:
    from gamut import exr, highbit
    from gamut import transfer as transfer_curves
    from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace


def _input_path(path):
    if folder_paths is not None and not os.path.isabs(path):
        return folder_paths.get_annotated_filepath(path)
    return path


def _save_batch(image, prefix, extension, writer, start_frame=1001, frame_pad=4, version=1):
    if image.ndim != 4 or image.shape[-1] not in (3, 4) or 0 in image.shape:
        raise ValueError("Saving requires a nonempty IMAGE batch [B, H, W, 3 or 4].")
    if start_frame < 0 or frame_pad < 1 or version < 1:
        raise ValueError("start_frame must be nonnegative; frame_pad and version must be positive.")
    prefix = os.fspath(prefix)
    if not prefix or prefix.endswith(("/", "\\")):
        raise ValueError("Supply a filename prefix, not an empty name or directory.")
    if os.path.isabs(prefix):
        folder, name = Path(prefix).parent, Path(prefix).name
    else:
        if folder_paths is None:
            raise RuntimeError("Relative output prefixes require ComfyUI; use an absolute prefix.")
        folder, name, _, _, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), image.shape[2], image.shape[1]
        )
        folder = Path(folder)
    paths = [
        folder / f"{name}_v{version:03d}.{start_frame + index:0{frame_pad}d}.{extension}"
        for index in range(image.shape[0])
    ]
    # Preflight the whole batch before writing any frame.
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    folder.mkdir(parents=True, exist_ok=True)
    for frame, path in zip(image, paths):
        # Exclusive creation also refuses a destination created after preflight.
        with path.open("xb"):
            pass
        try:
            writer(path, frame)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return {"ui": {"text": [str(path) for path in paths]}, "result": ()}


class GamutLoadEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "fallback_primaries": (list(PRIMARIES), {"default": "rec709"}),
                "fallback_transfer": (list(TRANSFERS), {"default": "linear"}),
                "tonemap_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Display preview: output becomes clipped Rec.709/sRGB. Keep off for HDR export.",
                }),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "COLORSPACE")
    RETURN_NAMES = ("image", "mask", "colorspace")
    FUNCTION = "load"
    CATEGORY = "Gamut/IO"

    def load(self, path, fallback_primaries="rec709", fallback_transfer="linear",
             tonemap_preview=False, colorspace=None):
        return exr.load_image(
            _input_path(path), fallback_primaries, fallback_transfer,
            tonemap_preview, colorspace,
        )

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        stat = Path(_input_path(path)).stat()
        return stat.st_mtime_ns, stat.st_size


class GamutSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "Gamut/render"}),
                "primaries": (list(PRIMARIES), {"default": "rec709"}),
                "transfer": (list(TRANSFERS), {"default": "linear"}),
                "half": ("BOOLEAN", {"default": True}),
                "compression": (list(exr._COMPRESSIONS), {"default": "zip"}),
                "start_frame": ("INT", {"default": 1001, "min": 0}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 12}),
                "version": ("INT", {"default": 1, "min": 1}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/IO"
    DESCRIPTION = "Write tagged float EXR frames. Widgets declare values; they do not convert pixels."

    def save(self, image, prefix="Gamut/render", primaries="rec709", transfer="linear",
             half=True, compression="zip", start_frame=1001, frame_pad=4, version=1,
             colorspace=None):
        cs = colorspace or ColorSpace(primaries, transfer, "widget")
        writer = partial(exr.write_exr, colorspace=cs, half=half, compression=compression)
        return _save_batch(image.detach().cpu().numpy(), prefix, "exr", writer,
                           start_frame, frame_pad, version)


class GamutSaveHighBit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "Gamut/image"}),
                "format": (["png16", "tiff16"], {"default": "png16"}),
                "transfer": (list(TRANSFERS), {"default": "srgb"}),
                "dither": ("BOOLEAN", {"default": False}),
            },
            "optional": {"colorspace": ("COLORSPACE",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/IO"

    def save(self, image, prefix="Gamut/image", format="png16", transfer="srgb",
             dither=False, colorspace=None):
        if format not in ("png16", "tiff16"):
            raise ValueError(f"Unknown high-bit format: {format!r}")
        source_transfer = colorspace.transfer if colorspace is not None else "linear"
        pixels = image
        if source_transfer != transfer:
            pixels = transfer_curves.transform_rgb(pixels, source_transfer, "decode")
            pixels = transfer_curves.transform_rgb(pixels, transfer, "encode")
        writer = highbit.save_png16 if format == "png16" else highbit.save_tiff16
        return _save_batch(pixels, prefix, "png" if format == "png16" else "tif",
                           partial(writer, dither=dither))
