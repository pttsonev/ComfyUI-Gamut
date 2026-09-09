"""ComfyUI contracts, standalone registration, and real node IO regressions."""

import ast
import importlib.util
import inspect
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import OpenEXR
import pytest
import torch

from gamut import exr, ocio_bridge
from gamut.colorspace import ColorSpace
from nodes import core_nodes, info_nodes, io_nodes, ltx_nodes, ocio_nodes
from tests.conftest import DEVICES


ROOT = Path(__file__).resolve().parents[1]
CORE_CLASSES = [
    core_nodes.GamutColorSpaceConvert, core_nodes.GamutTransfer,
    io_nodes.GamutLoadEXR, io_nodes.GamutSaveEXR, io_nodes.GamutSaveHighBit,
    ltx_nodes.GamutLTXHDRDecode, info_nodes.GamutColorSpaceInfo,
]
OCIO_CLASSES = [ocio_nodes.GamutOCIOTransform, ocio_nodes.GamutOCIODisplayView, ocio_nodes.GamutApplyLUT]
ALL_CLASSES = CORE_CLASSES + OCIO_CLASSES
EXPECTED_RETURNS = {
    "GamutColorSpaceConvert": ("IMAGE", "COLORSPACE"),
    "GamutTransfer": ("IMAGE", "COLORSPACE"),
    "GamutLoadEXR": ("IMAGE", "MASK", "COLORSPACE"),
    "GamutSaveEXR": (), "GamutSaveHighBit": (),
    "GamutLTXHDRDecode": ("IMAGE", "COLORSPACE"),
    "GamutColorSpaceInfo": ("STRING",),
    "GamutOCIOTransform": ("IMAGE", "COLORSPACE"),
    "GamutOCIODisplayView": ("IMAGE",), "GamutApplyLUT": ("IMAGE",),
}


@pytest.fixture(scope="module")
def package():
    name = "gamut_node_test_package"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


@pytest.mark.parametrize("node", ALL_CLASSES, ids=lambda node: node.__name__)
def test_every_node_has_a_consistent_comfyui_contract(node):
    schema = node.INPUT_TYPES()
    assert isinstance(schema, dict) and "required" in schema
    assert node.RETURN_TYPES == EXPECTED_RETURNS[node.__name__]
    assert node.CATEGORY.startswith("Gamut/")
    method = getattr(node(), node.FUNCTION)
    assert callable(method)
    parameters = inspect.signature(method).parameters
    for name, field in {**schema["required"], **schema.get("optional", {})}.items():
        assert name in parameters
        assert isinstance(field, tuple) and len(field) in (1, 2)
        assert isinstance(field[0], (str, list))
        if isinstance(field[0], list):
            assert field[0]
            if len(field) == 2 and "default" in field[1]:
                assert field[1]["default"] in field[0]
    assert schema["optional"]["colorspace"] == ("COLORSPACE",)
    if hasattr(node, "RETURN_NAMES"):
        assert len(node.RETURN_NAMES) == len(node.RETURN_TYPES)
    if not node.RETURN_TYPES:
        assert node.OUTPUT_NODE is True


def test_mappings_are_complete_for_current_capability_and_custom_type_is_frozen(package):
    names = {node.__name__ for node in CORE_CLASSES}
    if package.ocio_bridge.available():
        names.update(node.__name__ for node in OCIO_CLASSES)
    assert set(package.NODE_CLASS_MAPPINGS) == names
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == names
    assert all(name.startswith("Gamut: ") for name in package.NODE_DISPLAY_NAME_MAPPINGS.values())
    assert len(set(package.NODE_DISPLAY_NAME_MAPPINGS.values())) == len(names)
    assert package.CUSTOM_TYPES["COLORSPACE"].__dataclass_params__.frozen


@pytest.mark.parametrize("available", [False, True])
def test_registration_is_gated_only_by_optional_capability(package, monkeypatch, available):
    monkeypatch.setattr(package.ocio_bridge, "available", lambda: available)
    package.__spec__.loader.exec_module(package)
    expected = {node.__name__ for node in CORE_CLASSES + (OCIO_CLASSES if available else [])}
    assert set(package.NODE_CLASS_MAPPINGS) == expected
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == expected
    for name, node in package.NODE_CLASS_MAPPINGS.items():
        assert name == node.__name__


def test_ocio_disabled_reason_is_logged_once_on_reload(package, monkeypatch, caplog):
    monkeypatch.setattr(package.ocio_bridge, "available", lambda: False)
    monkeypatch.setattr(package, "_OCIO_NOTICE_EMITTED", False, raising=False)
    with caplog.at_level(logging.INFO, logger=package.__name__):
        package.__spec__.loader.exec_module(package)
        package.__spec__.loader.exec_module(package)
    messages = [record.message for record in caplog.records if "OCIO nodes disabled" in record.message]
    assert len(messages) == 1
    assert "opencolorio" in messages[0]


def test_package_imports_without_ocio_or_comfyui_and_with_host_nodes_name_occupied():
    script = """
import importlib.abc
import importlib.util
import pathlib
import sys
import types

class MissingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('folder_paths', 'comfy', 'PyOpenColorIO', 'opencolorio'):
            raise ModuleNotFoundError('Dependency deliberately absent: ' + fullname)
        return None

sys.meta_path.insert(0, MissingDependencies())
sys.modules['nodes'] = types.ModuleType('nodes')  # ComfyUI has its own nodes module.
spec = importlib.util.spec_from_file_location('gamut_custom_node', pathlib.Path('__init__.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert len(module.NODE_CLASS_MAPPINGS) == 7
assert set(module.NODE_CLASS_MAPPINGS) == set(module.NODE_DISPLAY_NAME_MAPPINGS)
assert 'PyOpenColorIO' not in sys.modules
for node in module.NODE_CLASS_MAPPINGS.values():
    assert isinstance(node.INPUT_TYPES(), dict)
print('standalone package import passed')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "standalone package import passed" in result.stdout


def test_node_classes_contain_no_colour_arithmetic():
    for node in ALL_CLASSES:
        tree = ast.parse(inspect.getsource(node))
        assert not any(isinstance(item, ast.BinOp) for item in ast.walk(tree)), node.__name__
    for path in (ROOT / "nodes").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                assert not any(name.name in ("torch", "numpy", "cv2", "PyOpenColorIO") for name in item.names)


@pytest.mark.parametrize("node", [ocio_nodes.GamutOCIOTransform, ocio_nodes.GamutOCIODisplayView])
def test_custom_config_string_connections_still_validate_image_types(node):
    assert node.VALIDATE_INPUTS(input_types={
        "image": "IMAGE", "src": "STRING", "colorspace": "COLORSPACE",
    }) is True
    assert "requires IMAGE" in node.VALIDATE_INPUTS(input_types={"image": "LATENT"})


@pytest.mark.parametrize("device", DEVICES)
def test_core_nodes_honour_wired_source_and_preserve_alpha(device):
    image = torch.tensor([[[-0.1, 0.2, 2.0, 0.37]]], device=device).unsqueeze(0)
    tagged = ColorSpace("acescg", "linear", "exr:chromaticities")
    converted, cs = core_nodes.GamutColorSpaceConvert().convert(image, "rec709", "acescg", tagged)
    assert converted is image  # Wired ACEScg overrides the Rec.709 widget.
    assert cs.primaries == "acescg" and cs.transfer == "linear"
    encoded, encoded_cs = core_nodes.GamutTransfer().apply(image, "encode", "g22", colorspace=tagged)
    assert torch.equal(encoded[..., 3:], image[..., 3:])
    decoded, decoded_cs = core_nodes.GamutTransfer().apply(
        encoded, "decode", "srgb", colorspace=encoded_cs
    )
    torch.testing.assert_close(decoded, image, atol=1e-6, rtol=0)
    assert decoded_cs.primaries == "acescg" and decoded_cs.transfer == "linear"


def test_non_linear_primaries_and_double_encoding_are_rejected():
    image = torch.ones((1, 2, 3, 3))
    cs = ColorSpace("rec709", "logc3", "widget")
    with pytest.raises(ValueError, match="Gamut: Transfer"):
        core_nodes.GamutColorSpaceConvert().convert(image, colorspace=cs)
    with pytest.raises(ValueError, match="decode"):
        core_nodes.GamutTransfer().apply(image, "encode", "srgb", colorspace=cs)


def test_untagged_transfer_uses_widget_primaries_and_curve():
    result, cs = core_nodes.GamutTransfer().apply(torch.full((1, 1, 1, 3), 0.5),
                                               "decode", "logc3", "rec2020")
    torch.testing.assert_close(result, torch.full_like(result, 0.513383), atol=1e-6, rtol=0)
    assert cs.primaries == "rec2020" and cs.transfer == "linear"


def test_load_exr_returns_rgb_mask_and_real_tags_before_fallbacks(tmp_path):
    path = tmp_path / "tagged.exr"
    pixels = np.array([[[-0.5, 0.5, 100.0, 0.25]]], dtype=np.float32)
    exr.write_exr(path, pixels, ColorSpace("acescg", "linear", "widget"), half=False)
    image, mask, cs = io_nodes.GamutLoadEXR().load(
        str(path), "ap0", "g24", colorspace=ColorSpace("rec2020", "srgb", "widget")
    )
    np.testing.assert_array_equal(image.numpy(), pixels[None, ..., :3])
    torch.testing.assert_close(mask, torch.tensor([[[0.75]]]), atol=0, rtol=0)
    assert cs == ColorSpace("acescg", "linear", "exr:chromaticities")


def test_untagged_load_records_fallback_and_has_zero_mask(tmp_path):
    path = tmp_path / "untagged.exr"
    with OpenEXR.File({}, {"RGB": np.full((2, 3, 3), 0.5, dtype=np.float32)}) as raw:
        raw.write(str(path))
    _, mask, cs = io_nodes.GamutLoadEXR().load(str(path), "ap0", "g22")
    assert cs == ColorSpace("ap0", "g22", "widget:fallback")
    assert mask.shape == (1, 2, 3) and torch.count_nonzero(mask) == 0
    wired = ColorSpace("rec2020", "linear", "widget")
    assert io_nodes.GamutLoadEXR().load(str(path), colorspace=wired)[2] is wired


def test_preview_is_explicit_and_changes_the_returned_declaration(tmp_path):
    path = tmp_path / "hdr.exr"
    exr.write_exr(path, np.array([[[-0.5, 0.5, 100.0]]], np.float32),
                  ColorSpace("rec709", "linear", "widget"), half=False)
    original, _, cs = io_nodes.GamutLoadEXR().load(str(path))
    preview, _, preview_cs = io_nodes.GamutLoadEXR().load(str(path), tonemap_preview=True)
    assert original.min() < 0 and original.max() > 1 and cs.transfer == "linear"
    assert preview.min() >= 0 and preview.max() <= 1
    assert (preview_cs.primaries, preview_cs.transfer) == ("rec709", "srgb")


def test_relative_save_uses_comfy_paths_version_frames_and_canonical_tags(tmp_path, monkeypatch):
    calls = []

    def save_path(prefix, output_dir, width, height):
        calls.append((prefix, output_dir, width, height))
        return str(tmp_path / "shots"), "beauty", 93, "shots", prefix

    monkeypatch.setattr(io_nodes, "folder_paths", SimpleNamespace(
        get_output_directory=lambda: str(tmp_path), get_save_image_path=save_path,
    ))
    image = torch.linspace(-0.25, 100., 48).reshape(2, 2, 3, 4)
    cs = ColorSpace("acescg", "linear", "exr:chromaticities")
    result = io_nodes.GamutSaveEXR().save(image, "shots/beauty", half=False, colorspace=cs)
    assert calls == [("shots/beauty", str(tmp_path), 3, 2)]
    paths = [tmp_path / "shots" / f"beauty_v001.{frame}.exr" for frame in (1001, 1002)]
    assert result["ui"]["text"] == [str(path) for path in paths]
    for index, path in enumerate(paths):
        pixels, read_cs, header = exr.read_exr(path)
        np.testing.assert_array_equal(pixels, image[index].numpy())
        assert read_cs.primaries == "acescg"
        assert header["colorSpace"] == exr._CANONICAL_NAMES[("acescg", "linear")]


def test_absolute_save_needs_no_comfy_and_uses_requested_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(io_nodes, "folder_paths", None)
    image = torch.ones((1, 2, 3, 3))
    prefix = str(tmp_path / "nested" / "shot")
    io_nodes.GamutSaveEXR().save(image, prefix, start_frame=7, frame_pad=6, version=12)
    assert (tmp_path / "nested" / "shot_v012.000007.exr").exists()
    with pytest.raises(RuntimeError, match="Relative output prefixes"):
        io_nodes.GamutSaveEXR().save(image, "relative/shot")


def test_existing_later_frame_refuses_the_entire_batch_before_any_write(tmp_path):
    protected = tmp_path / "shot_v001.1002.exr"
    protected.write_bytes(b"existing render")
    with pytest.raises(FileExistsError, match="overwrite"):
        io_nodes.GamutSaveEXR().save(torch.ones((2, 2, 3, 3)), str(tmp_path / "shot"))
    assert protected.read_bytes() == b"existing render"
    assert not (tmp_path / "shot_v001.1001.exr").exists()


def test_file_created_after_preflight_is_also_preserved(tmp_path, monkeypatch):
    protected = tmp_path / "race_v001.1001.exr"
    original_open = Path.open

    def racing_open(path, mode="r", *args, **kwargs):
        if path == protected and mode == "xb":
            with original_open(path, "wb") as output:
                output.write(b"concurrent render")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        io_nodes.GamutSaveEXR().save(torch.ones((1, 2, 3, 3)), str(tmp_path / "race"))
    assert protected.read_bytes() == b"concurrent render"


def test_writer_error_propagates_and_removes_its_own_incomplete_frame(tmp_path, monkeypatch):
    def fail(path, pixels, **kwargs):
        path.write_bytes(b"incomplete")
        raise RuntimeError("writer failed")

    monkeypatch.setattr(exr, "write_exr", fail)
    with pytest.raises(RuntimeError, match="writer failed"):
        io_nodes.GamutSaveEXR().save(torch.ones((1, 2, 3, 3)), str(tmp_path / "failed"))
    assert not (tmp_path / "failed_v001.1001.exr").exists()


@pytest.mark.parametrize("format", ["png16", "tiff16"])
def test_highbit_node_encodes_once_and_forwards_dither_and_alpha(tmp_path, monkeypatch, format):
    received = []

    def record(path, image, dither=False):
        received.append((image.clone(), dither))

    monkeypatch.setattr(io_nodes.highbit, "save_png16" if format == "png16" else "save_tiff16", record)
    image = torch.tensor([[[[0.21404114, 0.21404114, 0.21404114, 0.37]]]])
    node = io_nodes.GamutSaveHighBit()
    node.save(image, str(tmp_path / "encoded"), format, "srgb", True)
    expected = image[0].clone()
    expected[..., :3] = 0.5
    torch.testing.assert_close(received[0][0], expected, atol=1e-6, rtol=0)
    assert received[0][1] is True
    encoded = received[0][0].unsqueeze(0)
    node.save(encoded, str(tmp_path / "already"), format, "srgb", False,
              ColorSpace("rec709", "srgb", "widget"))
    torch.testing.assert_close(received[1][0], encoded[0], atol=0, rtol=0)


def test_ltx_node_delegates_hdr_decode_and_rejects_a_linear_input_tag():
    image = torch.ones((1, 1, 1, 3))
    result, cs = ltx_nodes.GamutLTXHDRDecode().decode(image, "acescg", 1.0)
    assert result.min() > 100 and cs.primaries == "acescg" and cs.transfer == "linear"
    with pytest.raises(ValueError, match="VAE codes"):
        ltx_nodes.GamutLTXHDRDecode().decode(image, colorspace=cs)


def test_info_reports_provenance_and_unknown_chromaticities_verbatim(tmp_path):
    path = tmp_path / "unknown.exr"
    with OpenEXR.File({"chromaticities": (0.1,) * 8},
                     {"RGB": np.zeros((1, 1, 3), np.float32)}) as raw:
        raw.write(str(path))
    cs = ColorSpace("rec709", "linear", "widget:fallback")
    result = info_nodes.GamutColorSpaceInfo().inspect(path=str(path), colorspace=cs)
    text = result["result"][0]
    assert "widget:fallback" in text and "untagged or unknown" in text
    assert repr(exr.read_exr(path)[2]["chromaticities"]) in text
    assert result["ui"]["text"] == [text]


@pytest.mark.skipif(not ocio_bridge.available(), reason="Optional opencolorio is not installed")
def test_ocio_node_resolves_wired_canonical_source_and_output_declaration():
    image = torch.ones((1, 2, 3, 3))
    cs = ColorSpace("rec709", "linear", "widget")
    result, output_cs = ocio_nodes.GamutOCIOTransform().transform(
        image, ocio_bridge.DEFAULT_CONFIG, "invalid widget source",
        exr._CANONICAL_NAMES[("acescg", "linear")], cs,
    )
    assert result.shape == image.shape and result.dtype == torch.float32
    assert (output_cs.primaries, output_cs.transfer) == ("acescg", "linear")
    cfg = ocio_bridge.get_config()
    assert ocio_bridge.source_name(output_cs, cfg) == output_cs.source[len("ocio:"):]


@pytest.mark.skipif(not ocio_bridge.available(), reason="Optional opencolorio is not installed")
def test_lut_node_forwards_direction_and_interpolation(tmp_path):
    path = tmp_path / "identity.cube"
    path.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n", encoding="ascii")
    image = torch.tensor([[[[0.1, 0.2, 0.3, 0.7]]]])
    result, = ocio_nodes.GamutApplyLUT().apply(image, str(path), "forward", "linear")
    torch.testing.assert_close(result, image, atol=1e-6, rtol=0)
