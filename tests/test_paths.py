"""Golden filenames, directory side effects, overwrite protection and sequence syntax."""

import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from gamut import paths
from nodes import io_nodes


EXR_SAVERS = [io_nodes.GamutSaveEXR, io_nodes.GamutSaveDataEXR, io_nodes.GamutSaveMultilayerEXR]


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    calls = []

    def save_path(prefix, output, width, height):
        calls.append((prefix, width, height))
        folder = Path(output) / Path(prefix).parent
        folder.mkdir(parents=True, exist_ok=True)  # real ComfyUI helper's implicit side effect
        name = Path(prefix).name
        numbers = [int(match.group(1)) for file in folder.iterdir()
                   if (match := re.match(re.escape(name) + r"_(\d+)_\.", file.name))]
        return str(folder), name, max(numbers, default=0) + 1, str(Path(prefix).parent), prefix

    monkeypatch.setattr(io_nodes, "folder_paths", SimpleNamespace(
        get_output_directory=lambda: str(tmp_path), get_save_image_path=save_path,
        get_annotated_filepath=lambda path: str(tmp_path / path.removesuffix(" [output]")),
    ))
    return calls


@pytest.mark.parametrize("node", EXR_SAVERS)
@pytest.mark.parametrize("mode,layout,prefix,expected", [
    ("versioned", "suffix", "shots/beauty", "shots/beauty_v012.000007.exr"),
    ("versioned", "directory", "shots/beauty", "shots/v012/beauty.000007.exr"),
    ("versioned", "none", "shots/beauty", "shots/beauty.000007.exr"),
    ("filename", "suffix", "shots/{version}/beauty.{frame}.exr", "shots/v012/beauty.000007.exr"),
    ("filename", "directory", "shots/custom.EXR", "shots/custom.EXR"),
    ("hq", "suffix", "shots/beauty", "shots/beauty_00001_.exr"),
])
def test_relative_names_and_preflight_agree(node, mode, layout, prefix, expected, comfy, tmp_path):
    options = dict(prefix=prefix, path_mode=mode, version_layout=layout, start_frame=7, frame_pad=6, version=12)
    assert node.VALIDATE_INPUTS(**options) is True
    assert not list(tmp_path.iterdir())  # validation must not create the missing directory
    result = node().save(torch.ones(1, 1, 2, 3), **options)
    assert result["ui"]["text"] == [str(tmp_path / expected)]
    if mode != "hq":
        assert node.VALIDATE_INPUTS(**options) is not True


@pytest.mark.parametrize("mode,layout,expected", [
    ("versioned", "suffix", "shot_v001.1001.exr"),
    ("hq", "suffix", "shot_v001.1001.exr"),
    ("hq", "directory", "v001/shot.1001.exr"),
    ("hq", "none", "shot.1001.exr"),
])
def test_absolute_modes_bypass_comfy(mode, layout, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(io_nodes, "folder_paths", None)
    result = io_nodes.GamutSaveEXR().save(torch.ones(1, 1, 1, 3), str(tmp_path / "shot"),
                                        path_mode=mode, version_layout=layout)
    assert result["ui"]["text"] == [str(tmp_path / expected)]


@pytest.mark.parametrize("node", EXR_SAVERS)
@pytest.mark.parametrize("absolute", [False, True])
def test_disabled_mkdir_has_no_side_effects(node, absolute, comfy, tmp_path):
    prefix = str(tmp_path / "missing/shot") if absolute else "missing/shot"
    verdict = node.VALIDATE_INPUTS(prefix=prefix, create_path_if_missing=False)
    assert "Directory does not exist" in verdict
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        node().save(torch.ones(1, 1, 1, 3), prefix, create_path_if_missing=False)
    assert not list(tmp_path.iterdir()) and comfy == []


@pytest.mark.parametrize("node", EXR_SAVERS)
@pytest.mark.parametrize("token", ["%width%", "%second%"])
def test_template_filename_checks_static_parent_before_deferring(node, token, comfy, tmp_path):
    prefix = f"missing/beauty_{token}"
    verdict = node.VALIDATE_INPUTS(prefix=prefix, create_path_if_missing=False)
    assert isinstance(verdict, str)
    assert "Directory does not exist" in verdict
    assert str(tmp_path / "missing") in verdict
    assert "create_path_if_missing" in verdict
    assert not list(tmp_path.iterdir()) and comfy == []

    (tmp_path / "missing").mkdir()
    # Do not check a made-up filename with width=0 or the current clock value.
    assert node.VALIDATE_INPUTS(prefix=prefix, create_path_if_missing=False) is True
    assert list((tmp_path / "missing").iterdir()) == [] and comfy == []


@pytest.mark.parametrize("node", EXR_SAVERS)
def test_template_preflight_checks_syntax_and_known_ancestors(node, comfy, tmp_path):
    verdict = node.VALIDATE_INPUTS(prefix="missing/%width%/beauty", create_path_if_missing=False)
    assert isinstance(verdict, str) and "Directory does not exist" in verdict
    assert str(tmp_path / "missing") in verdict
    (tmp_path / "missing").mkdir()
    assert node.VALIDATE_INPUTS(prefix="missing/%width%/beauty", create_path_if_missing=False) is True
    verdict = node.VALIDATE_INPUTS(
        prefix="missing/beauty_%width%", version_layout="directory", create_path_if_missing=False,
    )
    assert isinstance(verdict, str) and str(tmp_path / "missing/v001") in verdict

    for prefix, options, message in (
        ("missing/beauty_%width%/", {}, "directory"),
        ("missing/beauty_%width%.png", {"path_mode": "filename"}, "extension"),
        ("../outside/beauty_%width%", {}, "outside the output folder"),
    ):
        verdict = node.VALIDATE_INPUTS(prefix=prefix, **options)
        assert isinstance(verdict, str) and message in verdict
    (tmp_path / "not_a_directory").write_bytes(b"existing file")
    verdict = node.VALIDATE_INPUTS(prefix="not_a_directory/beauty_%width%")
    assert isinstance(verdict, str) and "not a directory" in verdict
    assert comfy == []


def test_creation_flag_checks_final_version_directory(comfy, tmp_path):
    (tmp_path / "shots").mkdir()
    verdict = io_nodes.GamutSaveEXR.VALIDATE_INPUTS(
        prefix="shots/beauty", version_layout="directory", create_path_if_missing=False,
    )
    assert "v001" in verdict
    assert comfy == [] and not (tmp_path / "shots/v001").exists()


def test_hq_uses_counter_once_and_advances_only_on_executing_save(comfy, tmp_path):
    (tmp_path / "shot_00092_.exr").write_bytes(b"prior")
    node = io_nodes.GamutSaveEXR()
    assert not hasattr(node, "IS_CHANGED")  # do not defeat ComfyUI's unchanged-prompt cache
    first = node.save(torch.ones(2, 1, 1, 3), "shot", path_mode="hq", version=17, start_frame=900,
                      frame_pad=8, version_layout="directory")
    second = node.save(torch.ones(1, 1, 1, 3), "shot", path_mode="hq")
    assert [Path(path).name for path in first["ui"]["text"]] == ["shot_00093_.exr", "shot_00094_.exr"]
    assert [Path(path).name for path in second["ui"]["text"]] == ["shot_00095_.exr"]
    assert len(comfy) == 2
    assert (tmp_path / "shot_00092_.exr").read_bytes() == b"prior"


@pytest.mark.parametrize("node", EXR_SAVERS)
@pytest.mark.parametrize("mode", ["versioned", "filename", "hq"])
def test_later_collision_refuses_entire_batch(node, mode, tmp_path):
    prefix = str(tmp_path / ("shot.{frame}.exr" if mode == "filename" else "shot"))
    filename = "shot.1002.exr" if mode == "filename" else "shot_v001.1002.exr"
    protected = tmp_path / filename
    protected.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        node().save(torch.ones(2, 1, 1, 3), prefix, path_mode=mode)
    assert list(tmp_path.iterdir()) == [protected]
    assert protected.read_bytes() == b"keep"


@pytest.mark.parametrize("prefix,kwargs,match", [
    ("shot.exr", {"count": 2, "path_mode": "filename"}, "one frame"),
    ("shot.png", {"path_mode": "filename"}, "extension"),
    ("shot.{frame}.{frame}.exr", {"path_mode": "filename"}, "at most one"),
    ("shot.{unknown}.exr", {"path_mode": "filename"}, "unknown tokens"),
    ("{frame}/shot.exr", {"path_mode": "filename"}, "directory"),
    ("directory/", {}, "directory"),
    ("shot", {"frame_pad": 0}, "frame_pad"),
    ("shot", {"version": True}, "version"),
    ("shot", {"path_mode": "auto"}, "Unknown"),
])
def test_malformed_output_arguments(prefix, kwargs, match):
    with pytest.raises(ValueError, match=match):
        paths.prepare_output(prefix, "exr", **kwargs)


def test_no_duplicate_destinations_or_padding_truncation(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        paths.output_paths(tmp_path, "exact.exr", "exr", count=2, path_mode="filename")
    assert paths.frame_path(tmp_path, "s", "exr", 10001, 2, 1000).name == "s_v1000.10001.exr"


@pytest.mark.parametrize("pattern", ["shot.{frame}.exr", "shot.####.exr", "shot.%04d.exr"])
def test_explicit_sequences_share_numeric_order_and_stride(pattern):
    assert paths.sequence_paths(pattern, 7, 3, 2) == ["shot.0007.exr", "shot.0009.exr", "shot.0011.exr"]


@pytest.mark.parametrize("pattern", ["shot.{bad}.exr", "shot.##.%04d.exr", "shot.%d.exr",
                                     "shot.{frame}.{frame}.exr", "####/shot.exr", "shot.%099d.exr"])
def test_ambiguous_sequence_patterns_are_refused(pattern):
    with pytest.raises(ValueError):
        paths.sequence_paths(pattern)


def test_exact_sequence_file_and_invalid_range_controls():
    assert paths.sequence_paths("shot.exr") == ["shot.exr"]
    with pytest.raises(ValueError, match="frame_count"):
        paths.sequence_paths("shot.exr", frame_count=2)
    for key, value in (("start_frame", -1), ("frame_count", 0), ("frame_step", 0), ("frame_pad", 0)):
        with pytest.raises(ValueError, match=key):
            paths.sequence_paths("shot.####.exr", **{key: value})


def test_relative_containment_and_dynamic_prefix_preflight(comfy, tmp_path):
    verdict = io_nodes.GamutSaveEXR.VALIDATE_INPUTS(prefix="../escape/shot")
    assert "outside the output folder" in verdict
    assert comfy == []
    assert io_nodes.GamutSaveEXR.VALIDATE_INPUTS(prefix="%width%x%height%/shot") is True
    assert io_nodes.GamutSaveEXR.VALIDATE_INPUTS(prefix=None, path_mode=None) is True
    result = io_nodes.GamutSaveEXR().save(torch.zeros(1, 2, 3, 3), "%width%x%height%/shot")
    assert result["ui"]["text"] == [str(tmp_path / "3x2/shot_v001.1001.exr")]


@pytest.mark.skipif(os.name != "nt", reason="native Windows path semantics")
def test_native_windows_drive_and_unc_paths():
    assert paths.prepare_output(r"D:\show\shot", "exr") == r"D:\show\shot"
    assert paths.prepare_output(r"\\server\share\shot", "exr") == r"\\server\share\shot"
    with pytest.raises(ValueError, match="drive-relative"):
        paths.prepare_output("D:shot", "exr")


@pytest.mark.parametrize("node", EXR_SAVERS)
def test_new_filename_mode_preserves_a_racing_writer(tmp_path, monkeypatch, node):
    target = tmp_path / "manual.exr"
    original = Path.open
    def racing_open(path, mode="r", *args, **kwargs):
        if path == target and mode == "xb":
            with original(path, "wb") as file:
                file.write(b"owned by the concurrent writer")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        node().save(torch.ones(1, 1, 1, 3), str(target), path_mode="filename")
    assert target.read_bytes() == b"owned by the concurrent writer"


def test_relative_annotated_sequence_reads_resolve_each_frame(comfy, tmp_path):
    for frame in (1001, 1002):
        io_nodes.GamutSaveEXR().save(torch.ones(1, 1, 1, 3), f"shot.{frame}.exr", path_mode="filename")
    image, _, _ = io_nodes.GamutLoadEXR().load("shot.####.exr [output]", frame_count=2)
    assert image.shape == (2, 1, 1, 3)
