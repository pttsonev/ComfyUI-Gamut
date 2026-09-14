"""EXR and high-bit file nodes; ComfyUI path handling stays at this boundary."""

import os
import time
from functools import partial
from pathlib import Path

# ComfyUI's folder_paths is imported on first use, never at module load, so
# these nodes stay importable outside ComfyUI. Two states must stay distinct:
# _UNSET means "not yet attempted", None means "attempted and unavailable".
# Collapsing them would make a test that pins `folder_paths = None` to select
# the standalone path silently re-import the real module instead.
_UNSET = object()
folder_paths = _UNSET

if "." in __package__:
    from ..gamut import alpha, exr, exr_layers, highbit, paths as file_paths
    from ..gamut import transfer as transfer_curves
    from ..gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace
else:
    from gamut import alpha, exr, exr_layers, highbit, paths as file_paths
    from gamut import transfer as transfer_curves
    from gamut.colorspace import PRIMARIES, TRANSFERS, ColorSpace


def _get_folder_paths():
    """Return ComfyUI's folder_paths module, or None outside ComfyUI.

    The outcome is cached including the failure, so a missing ComfyUI costs one
    failed import rather than one per save.
    """
    global folder_paths
    if folder_paths is _UNSET:
        try:
            import folder_paths as module
        except ImportError:
            module = None
        folder_paths = module
    return folder_paths


def _input_path(path):
    if not os.path.isabs(path):
        path_service = _get_folder_paths()
        if path_service is not None:
            return path_service.get_annotated_filepath(path)
    return path


def _comfy_vars(prefix, width, height):
    now = time.localtime()
    values = dict(width=str(width), height=str(height), year=str(now.tm_year),
                  month=f"{now.tm_mon:02d}", day=f"{now.tm_mday:02d}",
                  hour=f"{now.tm_hour:02d}", minute=f"{now.tm_min:02d}", second=f"{now.tm_sec:02d}")
    for key, value in values.items():
        prefix = prefix.replace(f"%{key}%", value)
    return prefix


def _contained_parent(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if not parent.is_relative_to(root):
        raise ValueError("Saving image outside the output folder is not allowed.")


def _check_output_parent(folder, create_path_if_missing):
    if not folder.is_dir():
        if folder.exists() or os.path.lexists(folder):
            raise NotADirectoryError(f"Output parent is not a directory: {folder}")
        if not create_path_if_missing:
            raise FileNotFoundError(f"Directory does not exist: {folder}; enable create_path_if_missing.")


def _resolve_target(prefix, width=0, height=0, *, create_path_if_missing=True, preflight=False):
    """Resolve an already formatted prefix, guarding ComfyUI's implicit mkdir."""
    if os.path.isabs(prefix):
        folder, name, counter = Path(prefix).parent, Path(prefix).name, 1
    else:
        path_service = _get_folder_paths()
        if path_service is None:
            raise RuntimeError("Relative output prefixes require ComfyUI; use an absolute prefix.")
        prefix = _comfy_vars(prefix, width, height)
        root = path_service.get_output_directory()
        folder, name, counter = Path(root) / Path(prefix).parent, Path(prefix).name, 1
        _contained_parent(root, folder)
        _check_output_parent(folder, create_path_if_missing)
        if not folder.is_dir():
            if preflight:
                return folder, name, counter
            folder.mkdir(parents=True, exist_ok=True)
        folder, name, counter, _, _ = path_service.get_save_image_path(prefix, root, width, height)
        folder = Path(folder)
        _contained_parent(root, folder)
    _check_output_parent(folder, create_path_if_missing)
    return folder, name, counter


def _output_target(prefix, width=0, height=0):
    # Retain this small compatibility seam for existing callers and regression tests.
    prefix = file_paths.prepare_output(prefix, "exr")
    folder, name, _ = _resolve_target(prefix, width, height)
    return folder, name


def _frame_path(folder, name, extension, frame, frame_pad, version):
    return file_paths.frame_path(folder, name, extension, frame, frame_pad, version)


def _plan_output(prefix, extension, count=1, width=0, height=0, start_frame=1001,
                 frame_pad=4, version=1, path_mode="versioned", version_layout="suffix",
                 create_path_if_missing=True, preflight=False):
    if not isinstance(create_path_if_missing, bool):
        raise ValueError("create_path_if_missing must be a boolean.")
    prepared = file_paths.prepare_output(prefix, extension, count, start_frame, frame_pad,
                                         version, path_mode, version_layout)
    folder, name, counter = _resolve_target(
        prepared, width, height, create_path_if_missing=create_path_if_missing, preflight=preflight,
    )
    return file_paths.output_paths(folder, name, extension, count, start_frame, frame_pad,
                                   version, path_mode, version_layout,
                                   absolute=os.path.isabs(prefix), counter=counter)


def _refuse_existing_first_frame(prefix, extension, start_frame=1001, frame_pad=4, version=1,
                                 path_mode="versioned", version_layout="suffix", create_path_if_missing=True):
    """Queue-time early refusal; execution checks the full batch and races again."""
    if not isinstance(prefix, str) or not prefix:
        return True
    if any(value is None for value in (path_mode, version_layout, create_path_if_missing)):
        return True  # linked widget
    for value in (start_frame, frame_pad, version):
        if isinstance(value, bool) or not isinstance(value, int):
            return True
    if start_frame < 0 or frame_pad < 1 or version < 1:
        return True  # preserve existing argument-error behaviour
    try:
        if not os.path.isabs(prefix) and "%" in prefix:
            if not isinstance(create_path_if_missing, bool):
                raise ValueError("create_path_if_missing must be a boolean.")
            prepared = file_paths.prepare_output(
                prefix, extension, start_frame=start_frame, frame_pad=frame_pad,
                version=version, path_mode=path_mode, version_layout=version_layout,
            )
            path_service = _get_folder_paths()
            if path_service is not None:
                # A dynamic basename cannot change a known parent failure. If the
                # parent also varies, check its static ancestor without expanding
                # dimensions/clock values or calling ComfyUI's mkdir-capable helper.
                parent = Path(prepared).parent
                while "%" in str(parent):
                    parent = parent.parent
                root = path_service.get_output_directory()
                folder = Path(root) / parent
                _contained_parent(root, folder)
                _check_output_parent(folder, create_path_if_missing)
            return True  # only the unresolved destination checks are deferred
        first = _plan_output(prefix, extension, start_frame=start_frame, frame_pad=frame_pad,
                             version=version, path_mode=path_mode, version_layout=version_layout,
                             create_path_if_missing=create_path_if_missing, preflight=True)[0]
    except RuntimeError:
        return True  # relative prefix outside ComfyUI; execution explains the missing service
    except Exception as exc:  # ComfyUI's path service raises plain Exception for containment
        return str(exc)
    if os.path.lexists(first):
        return f"{first} already exists; bump version or change the prefix. Nothing was run."
    return True


def _save_batch(image, prefix, extension, writer, start_frame=1001, frame_pad=4, version=1,
                path_mode="versioned", version_layout="suffix", create_path_if_missing=True,
                frames=None):
    if image.ndim != 4 or image.shape[-1] not in (3, 4) or 0 in image.shape:
        raise ValueError("Saving requires a nonempty IMAGE batch [B, H, W, 3 or 4].")
    paths = _plan_output(prefix, extension, image.shape[0], image.shape[2], image.shape[1],
                         start_frame, frame_pad, version, path_mode, version_layout, create_path_if_missing)
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    folder = paths[0].parent
    if create_path_if_missing:
        folder.mkdir(parents=True, exist_ok=True)
    for frame, path in zip(image if frames is None else frames, paths):
        # Exclusive creation must succeed before entering cleanup: never unlink a race winner.
        with path.open("xb"):
            pass
        try:
            writer(path, frame)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return {"ui": {"text": [str(path) for path in paths]}, "result": ()}


def _path_inputs():
    return {
        "path_mode": (list(file_paths.PATH_MODES), {
            "default": "versioned", "tooltip": "Versioned prefix, HQ counter (relative paths), or exact filename/{frame} template.",
        }),
        "version_layout": (list(file_paths.VERSION_LAYOUTS), {"default": "suffix"}),
        "create_path_if_missing": ("BOOLEAN", {"default": True}),
    }


def _alpha_inputs():
    return {
        "alpha_mask": ("MASK", {"tooltip": "Transparency mask: opacity written as 1-mask. Replaces existing A."}),
        "alpha_image": ("IMAGE", {"tooltip": "Opacity IMAGE: selected channel is embedded unchanged. Replaces existing A."}),
        "alpha_channel": (["R", "G", "B", "A"], {"default": "R"}),
    }


def _sequence_inputs():
    return {
        "layer": ("STRING", {"default": "", "tooltip": "Exact layer name; empty selects root RGB(A)."}),
        "start_frame": ("INT", {"default": 1001, "min": 0}),
        "frame_count": ("INT", {"default": 1, "min": 1}),
        "frame_step": ("INT", {"default": 1, "min": 1}),
        "frame_pad": ("INT", {"default": 4, "min": 1, "max": 12}),
    }


def _read_paths(path, start_frame=1001, frame_count=1, frame_step=1, frame_pad=4):
    return [_input_path(filename) for filename in
            file_paths.sequence_paths(path, start_frame, frame_count, frame_step, frame_pad)]


def _colour_frames(image, opacity):
    for index, frame in enumerate(image):
        frame = frame.detach().cpu()
        if opacity is not None:
            frame = alpha.embed_alpha(frame, opacity[index].detach().cpu())
        yield exr.as_numpy(frame)


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
            "optional": {"colorspace": ("COLORSPACE",), **_sequence_inputs()},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "COLORSPACE")
    RETURN_NAMES = ("image", "mask", "colorspace")
    FUNCTION = "load"
    CATEGORY = "Gamut/IO"

    def load(self, path, fallback_primaries="rec709", fallback_transfer="linear",
             tonemap_preview=False, colorspace=None, layer="", start_frame=1001,
             frame_count=1, frame_step=1, frame_pad=4):
        return exr.load_sequence(
            _read_paths(path, start_frame, frame_count, frame_step, frame_pad),
            fallback_primaries, fallback_transfer, tonemap_preview, colorspace, layer,
        )

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        return _read_fingerprint(path, kwargs)


def _read_fingerprint(path, kwargs):
    controls = {name: kwargs[name] for name in ("start_frame", "frame_count", "frame_step", "frame_pad")
                if name in kwargs}
    return file_paths.fingerprint(_read_paths(path, **controls))


class GamutLoadDataEXR:
    @classmethod
    def INPUT_TYPES(cls):
        sequence = _sequence_inputs()
        sequence["layer"] = ("STRING", {"default": "normal", "tooltip": "Data layer; N reads older normal exports. Empty selects root components."})
        return {"required": {
            "path": ("STRING", {"default": ""}),
            "layer": sequence.pop("layer"),
            "components": (["XYZ", "RGB"], {"default": "XYZ"}),
        }, "optional": sequence}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "Gamut/IO"
    DESCRIPTION = "Read raw vector layers or sequences; never infer colour or encode normals."

    def load(self, path, layer="normal", components="XYZ", start_frame=1001,
             frame_count=1, frame_step=1, frame_pad=4):
        if components not in ("XYZ", "RGB"):
            raise ValueError("Data components must be XYZ or RGB.")
        return exr.load_sequence(_read_paths(path, start_frame, frame_count, frame_step, frame_pad),
                                 layer=layer, components=components)

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        return _read_fingerprint(path, kwargs)


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
            "optional": {"colorspace": ("COLORSPACE",), **_path_inputs(), **_alpha_inputs()},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/IO"
    DESCRIPTION = "Write tagged float EXR frames. Widgets declare values; they do not convert pixels."

    @classmethod
    def VALIDATE_INPUTS(cls, prefix=None, start_frame=1001, frame_pad=4, version=1,
                        path_mode="versioned", version_layout="suffix", create_path_if_missing=True):
        return _refuse_existing_first_frame(prefix, "exr", start_frame, frame_pad, version,
                                            path_mode, version_layout, create_path_if_missing)

    def save(self, image, prefix="Gamut/render", primaries="rec709", transfer="linear",
             half=True, compression="zip", start_frame=1001, frame_pad=4, version=1,
             colorspace=None, path_mode="versioned", version_layout="suffix", create_path_if_missing=True,
             alpha_mask=None, alpha_image=None, alpha_channel="R"):
        cs = colorspace or ColorSpace(primaries, transfer, "widget")
        exr.colour_header(cs, compression)
        opacity = alpha.extract_alpha(image, alpha_mask, alpha_image, alpha_channel)
        exr.validate_half_batch(_colour_frames(image, opacity), half)
        writer = partial(exr.write_exr, colorspace=cs, half=half, compression=compression)
        return _save_batch(image, prefix, "exr", writer, start_frame, frame_pad, version,
                           path_mode, version_layout, create_path_if_missing,
                           frames=_colour_frames(image, opacity))


class GamutSaveDataEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Three data channels. Files carry no colour metadata by design "
                               "(no chromaticities or colorSpace).",
                }),
                "prefix": ("STRING", {"default": "Gamut/data"}),
                "channel_naming": (["normals", "prefix"], {"default": "normals"}),
                "channel_prefix": ("STRING", {
                    "default": "N",
                    "tooltip": "Used with prefix naming to write <prefix>.X, <prefix>.Y, <prefix>.Z.",
                }),
                "half": ("BOOLEAN", {"default": True}),
                "compression": (list(exr._COMPRESSIONS), {"default": "zip"}),
                "start_frame": ("INT", {"default": 1001, "min": 0}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 12}),
                "version": ("INT", {"default": 1, "min": 1}),
            },
            "optional": _path_inputs(),
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/IO"
    DESCRIPTION = "Write data EXR frames with no colour metadata by design; values are unchanged."

    @classmethod
    def VALIDATE_INPUTS(cls, prefix=None, start_frame=1001, frame_pad=4, version=1,
                        path_mode="versioned", version_layout="suffix", create_path_if_missing=True):
        return _refuse_existing_first_frame(prefix, "exr", start_frame, frame_pad, version,
                                            path_mode, version_layout, create_path_if_missing)

    def save(self, image, prefix="Gamut/data", channel_naming="normals", channel_prefix="N",
             half=True, compression="zip", start_frame=1001, frame_pad=4, version=1,
             path_mode="versioned", version_layout="suffix", create_path_if_missing=True):
        if image.ndim != 4 or image.shape[-1] != 3 or 0 in image.shape:
            raise ValueError("Saving data requires a nonempty IMAGE batch [B, H, W, 3].")
        if channel_naming not in ("normals", "prefix"):
            raise ValueError(f"Unknown data channel naming: {channel_naming!r}")
        name = "N" if channel_naming == "normals" else channel_prefix
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("Supply a nonempty channel prefix without NUL.")
        names = [f"{name}.{axis}" for axis in "XYZ"]

        def writer(path, frame):
            frame = exr.as_numpy(frame)
            channels = {name: frame[..., index] for index, name in enumerate(names)}
            exr.write_data_exr(path, channels, half=half, compression=compression)

        exr.validate_half_batch(image, half, "Data")
        return _save_batch(image, prefix, "exr", writer, start_frame, frame_pad, version,
                           path_mode, version_layout, create_path_if_missing)


class GamutSaveMultilayerEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgb": ("IMAGE",),
            "prefix": ("STRING", {"default": "Gamut/render"}),
            "primaries": (list(PRIMARIES), {"default": "rec709"}),
            "transfer": (list(TRANSFERS), {"default": "linear"}),
            "rgb_half": ("BOOLEAN", {"default": True}),
            "albedo_half": ("BOOLEAN", {"default": True}),
            "irradiance_half": ("BOOLEAN", {"default": True}),
            "normal_half": ("BOOLEAN", {"default": False}),
            "compression": (list(exr._COMPRESSIONS), {"default": "zip"}),
            "start_frame": ("INT", {"default": 1001, "min": 0}),
            "frame_pad": ("INT", {"default": 4, "min": 1, "max": 12}),
            "version": ("INT", {"default": 1, "min": 1}),
        }, "optional": {
            "colorspace": ("COLORSPACE",),
            "albedo": ("IMAGE",), "irradiance": ("IMAGE",),
            "normal": ("IMAGE", {"tooltip": "Signed XYZ, unchanged. Decode [0,1] normals with Gamut: Normal Decode first."}),
            "rgb_colorspace": ("COLORSPACE",), "albedo_colorspace": ("COLORSPACE",),
            "irradiance_colorspace": ("COLORSPACE",),
            **_path_inputs(), **_alpha_inputs(),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Gamut/IO"
    DESCRIPTION = "One tagged EXR per frame with colour layers and raw normal.X/Y/Z. All colour layers must share one declaration."

    @classmethod
    def VALIDATE_INPUTS(cls, prefix=None, start_frame=1001, frame_pad=4, version=1,
                        path_mode="versioned", version_layout="suffix", create_path_if_missing=True):
        return _refuse_existing_first_frame(prefix, "exr", start_frame, frame_pad, version,
                                            path_mode, version_layout, create_path_if_missing)

    def save(self, rgb, prefix="Gamut/render", primaries="rec709", transfer="linear",
             rgb_half=True, albedo_half=True, irradiance_half=True, normal_half=False,
             compression="zip", start_frame=1001, frame_pad=4, version=1, colorspace=None,
             albedo=None, irradiance=None, normal=None, rgb_colorspace=None, albedo_colorspace=None,
             irradiance_colorspace=None, path_mode="versioned", version_layout="suffix",
             create_path_if_missing=True, alpha_mask=None, alpha_image=None, alpha_channel="R"):
        cs = colorspace or ColorSpace(primaries, transfer, "widget")
        storage = dict(rgb_half=rgb_half, albedo_half=albedo_half, irradiance_half=irradiance_half,
                       normal_half=normal_half, compression=compression)
        passes, opacity = exr_layers.prepare_batch(
            rgb, cs, albedo=albedo, irradiance=irradiance, normal=normal,
            rgb_colorspace=rgb_colorspace, albedo_colorspace=albedo_colorspace,
            irradiance_colorspace=irradiance_colorspace, alpha_mask=alpha_mask,
            alpha_image=alpha_image, alpha_channel=alpha_channel, **storage,
        )
        writer = partial(exr_layers.write_multilayer_exr, colorspace=cs, **storage)
        return _save_batch(rgb, prefix, "exr", writer, start_frame, frame_pad, version,
                           path_mode, version_layout, create_path_if_missing,
                           frames=exr_layers.batch_frames(passes, opacity))


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

    @classmethod
    def VALIDATE_INPUTS(cls, prefix=None, format="png16"):
        if format not in ("png16", "tiff16"):
            return True  # save() names the bad format itself
        return _refuse_existing_first_frame(prefix, "png" if format == "png16" else "tif")

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
