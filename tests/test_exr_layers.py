"""Real mixed-file IO and the colour/data boundary, including before-write refusals."""

import json

import numpy as np
import OpenEXR
import pytest
import torch

from gamut import exr, exr_layers
from gamut.colorspace import ColorSpace
from nodes.io_nodes import GamutSaveMultilayerEXR


CS = ColorSpace("acescg", "linear", "test")


def passes():
    return {
        "rgb": np.array([[[-0.5, 1.0 / 3.0, 64., 0.25], [0., 0.5, 2., 1.]]], np.float32),
        "albedo": np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], np.float32),
        "irradiance": np.array([[[1., 2., 4.], [8., 16., 32.]]], np.float32),
        "normal": np.array([[[-0.8, 0.1234567, -1.], [1., -0.1, 0.]]], np.float32),
    }


@pytest.mark.parametrize("compression,codec", [
    ("none", OpenEXR.NO_COMPRESSION), ("rle", OpenEXR.RLE_COMPRESSION),
    ("zips", OpenEXR.ZIPS_COMPRESSION), ("zip", OpenEXR.ZIP_COMPRESSION), ("piz", OpenEXR.PIZ_COMPRESSION),
])
def test_mixed_file_has_exact_channels_tags_types_and_unchanged_vectors(tmp_path, compression, codec):
    path = tmp_path / "mixed.exr"
    written = passes()
    exr_layers.write_multilayer_exr(path, written, CS, compression=compression)
    with OpenEXR.File(str(path), separate_channels=True) as file:
        assert len(file.parts) == 1
        assert file.header()["type"] == OpenEXR.scanlineimage
        assert file.header()["compression"] == codec
        assert file.header()["colorSpace"] == "ACEScg"
        np.testing.assert_allclose(file.header()["chromaticities"],
                                   (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767))
        assert json.loads(file.header()["gamut:layerRoles"]) == {
            "schema": 1, "colour": ["rgba", "albedo", "irradiance"], "data": ["normal"],
        }
        channels = file.channels()
        assert set(channels) == {"R", "G", "B", "A", "albedo.R", "albedo.G", "albedo.B",
                                 "irradiance.R", "irradiance.G", "irradiance.B", "normal.X", "normal.Y", "normal.Z"}
        for name, pixels in written.items():
            components = "XYZ" if name == "normal" else "RGBA"[:pixels.shape[-1]]
            dtype = np.float32 if name == "normal" else np.float16
            for i, component in enumerate(components):
                channel = component if name == "rgb" else f"{name}.{component}"
                assert channels[channel].pixels.dtype == dtype
                np.testing.assert_array_equal(channels[channel].pixels, pixels[..., i].astype(dtype))
    assert exr.channel_types(path)["normal.X"] == "FLOAT"
    assert exr.channel_types(path)["R"] == "HALF"


def test_independent_layer_precision_and_absent_passes(tmp_path):
    path = tmp_path / "select.exr"
    data = passes()
    data.pop("albedo")
    data["rgb"] = data["rgb"][..., :3]
    exr_layers.write_multilayer_exr(path, data, CS, rgb_half=False, irradiance_half=False, normal_half=True)
    with OpenEXR.File(str(path), separate_channels=True) as file:
        assert "A" not in file.channels() and "albedo.R" not in file.channels()
        assert file.channels()["R"].pixels.dtype == np.float32
        assert file.channels()["irradiance.B"].pixels.dtype == np.float32
        assert file.channels()["normal.X"].pixels.dtype == np.float16
        assert json.loads(file.header()["gamut:layerRoles"])["colour"] == ["rgba", "irradiance"]


def test_node_writes_one_frame_file_for_all_aligned_passes_and_alpha(tmp_path):
    data = {name: torch.from_numpy(value).unsqueeze(0).repeat(2, 1, 1, 1) for name, value in passes().items()}
    snapshots = {name: value.clone() for name, value in data.items()}
    mask = torch.tensor([[[0.2, 0.8]], [[0., 1.]]])
    result = GamutSaveMultilayerEXR().save(**data, colorspace=CS, alpha_mask=mask, prefix=str(tmp_path / "shot"))
    assert result == {"ui": {"text": [str(tmp_path / "shot_v001.1001.exr"), str(tmp_path / "shot_v001.1002.exr")]}, "result": ()}
    assert len(list(tmp_path.iterdir())) == 2
    for i, path in enumerate(result["ui"]["text"]):
        channels, _ = exr.read_data_exr(path)
        np.testing.assert_array_equal(channels["normal.Y"], data["normal"][i, ..., 1].numpy())
        np.testing.assert_array_equal(channels["A"], (1 - mask[i]).numpy().astype(np.float16).astype(np.float32))
    for name in data:
        torch.testing.assert_close(data[name], snapshots[name], atol=0, rtol=0)


@pytest.mark.parametrize("fault,match", [
    ("batch", "albedo expected"), ("height", "albedo expected"), ("channels", "albedo requires"),
    ("integer", "floating-point"), ("half", "irradiance.B"),
    ("space", "albedo_colorspace must match"), ("orphan", "without its image"),
    ("unknown_space", "Unknown EXR colour"), ("codec", "compression"),
])
def test_invalid_later_frames_or_declarations_create_nothing(tmp_path, fault, match):
    data = dict(rgb=torch.zeros(2, 1, 2, 3), albedo=torch.zeros(2, 1, 2, 3),
                irradiance=torch.zeros(2, 1, 2, 3), colorspace=CS)
    if fault == "batch": data["albedo"] = torch.zeros(1, 1, 2, 3)
    if fault == "height": data["albedo"] = torch.zeros(2, 2, 2, 3)
    if fault == "channels": data["albedo"] = torch.zeros(2, 1, 2, 4)
    if fault == "integer": data["albedo"] = data["albedo"].int()
    if fault == "half": data["irradiance"][1, 0, 0, 2] = 70000.
    if fault == "space": data["albedo_colorspace"] = ColorSpace("rec709", "linear", "test")
    if fault == "orphan":
        data["albedo"] = None
        data["albedo_colorspace"] = CS
    if fault == "unknown_space": data["colorspace"] = ColorSpace("unknown", "linear", "test")
    if fault == "codec": data["compression"] = "lossy"
    with pytest.raises((ValueError, TypeError), match=match):
        GamutSaveMultilayerEXR().save(**data, prefix=str(tmp_path / "missing" / "shot"))
    assert not list(tmp_path.iterdir())


def test_per_pass_provenance_can_differ_and_nonfinite_vectors_survive(tmp_path):
    normal = torch.tensor([[[[float("inf"), float("nan"), -float("inf")]]]])
    GamutSaveMultilayerEXR().save(torch.ones(1, 1, 1, 3), normal=normal,
        colorspace=CS, rgb_colorspace=ColorSpace("acescg", "linear", "widget"),
        prefix=str(tmp_path / "shot"))
    data, _ = exr.read_data_exr(tmp_path / "shot_v001.1001.exr")
    assert np.isposinf(data["normal.X"]).all() and np.isnan(data["normal.Y"]).all()
