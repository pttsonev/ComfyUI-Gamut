"""Optional real-OCIO integration checks; all skip when the binding is absent."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gamut import ocio_bridge


pytestmark = pytest.mark.skipif(
    not ocio_bridge.available(), reason="Optional opencolorio is not installed"
)


@pytest.fixture
def config():
    import PyOpenColorIO as ocio

    cfg = ocio.Config.CreateRaw()
    source = ocio.ColorSpace(name="source")
    destination = ocio.ColorSpace(name="destination")
    destination.setTransform(
        ocio.MatrixTransform(matrix=[2., 0., 0., 0., 0., 3., 0., 0.,
                                     0., 0., 4., 0., 0., 0., 0., 1.]),
        ocio.COLORSPACE_DIR_FROM_REFERENCE,
    )
    cfg.addColorSpace(source)
    cfg.addColorSpace(destination)
    cfg.setRole(ocio.ROLE_SCENE_LINEAR, "destination")
    cfg.addDisplayView("Test display", "Test view", "destination", "")
    return cfg


def test_available_is_cached(monkeypatch):
    calls = []
    real_import = importlib.import_module

    def record_import(name):
        calls.append(name)
        return real_import(name)

    ocio_bridge.available.cache_clear()
    try:
        monkeypatch.setattr(ocio_bridge.importlib, "import_module", record_import)
        assert ocio_bridge.available()
        assert ocio_bridge.available()
        assert calls == ["PyOpenColorIO"]
    finally:
        ocio_bridge.available.cache_clear()


def test_import_and_capability_probe_work_with_ocio_forced_absent():
    script = """
import importlib.abc
import sys

class BlockOCIO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PyOpenColorIO', 'opencolorio'):
            raise ModuleNotFoundError('OCIO deliberately absent')
        return None

sys.meta_path.insert(0, BlockOCIO())
from gamut import colorspace, primaries, transfer, exr, highbit, ltx, ocio_bridge
assert 'PyOpenColorIO' not in sys.modules
assert not ocio_bridge.available()
assert not ocio_bridge.available()
try:
    ocio_bridge.get_config()
except ImportError as exc:
    assert 'opencolorio' in str(exc)
else:
    raise AssertionError('Missing dependency must raise on use')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_default_and_named_builtin_configs():
    for spec in (ocio_bridge.DEFAULT_CONFIG, "ocio://" + ocio_bridge.DEFAULT_CONFIG):
        cfg = ocio_bridge.get_config(spec)
        names = ocio_bridge.list_spaces(cfg)
        assert "ACEScg" in names and "ACES2065-1" in names
    assert "ACEScg" in ocio_bridge.list_spaces(ocio_bridge.get_config())


def test_config_file_and_ocio_environment(tmp_path, monkeypatch, config):
    path = tmp_path / "test.ocio"
    path.write_text(config.serialize(), encoding="utf-8")
    monkeypatch.setenv("OCIO", str(path))
    for spec in (path, str(path), "$OCIO"):
        loaded = ocio_bridge.get_config(spec)
        assert "source" in ocio_bridge.list_spaces(loaded)
        assert "destination" in ocio_bridge.list_spaces(loaded)


def test_unset_ocio_environment_raises(monkeypatch):
    monkeypatch.delenv("OCIO", raising=False)
    with pytest.raises(ValueError, match="unset"):
        ocio_bridge.get_config("$OCIO")


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 3), (2, 3, 4, 4)])
def test_transform_packed_copy_handles_rgb_rgba_and_batches(config, shape):
    pixels = np.linspace(-0.5, 2., np.prod(shape), dtype=np.float64).reshape(shape)[..., ::-1]
    original = pixels.copy()
    result = ocio_bridge.apply_transform(pixels, config, "source", "destination")
    scale = np.array([2, 3, 4, 1][:shape[-1]], dtype=np.float32)
    np.testing.assert_allclose(result, pixels.astype(np.float32) * scale, atol=1e-6, rtol=0)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert result.shape == pixels.shape
    assert not np.shares_memory(result, pixels)
    np.testing.assert_array_equal(pixels, original)


def test_display_uses_explicit_source_instead_of_scene_linear_role(config):
    pixels = np.array([[[0.1, 0.2, 0.3, 0.5]]], dtype=np.float32)
    result = ocio_bridge.apply_display(pixels, config, "source", "Test display", "Test view")
    np.testing.assert_allclose(result, [[[0.2, 0.6, 1.2, 0.5]]], atol=1e-6, rtol=0)
    # The deliberately different scene_linear role would incorrectly give identity.
    assert not np.allclose(result[..., :3], pixels[..., :3])


@pytest.mark.parametrize("src", [None, "", "  "])
def test_display_rejects_an_empty_source(config, src):
    with pytest.raises(ValueError, match="explicit source"):
        ocio_bridge.apply_display(np.ones(3), config, src, "Test display", "Test view")


def test_display_has_no_implicit_source_argument(config):
    with pytest.raises(TypeError):
        ocio_bridge.apply_display(np.ones(3), config, display="Test display", view="Test view")


def test_file_lut_forward_and_inverse(tmp_path):
    path = tmp_path / "scale.cube"
    path.write_text(
        'TITLE "half scale"\nLUT_1D_SIZE 2\nDOMAIN_MIN 0 0 0\n'
        'DOMAIN_MAX 1 1 1\n0 0 0\n0.5 0.5 0.5\n', encoding="ascii",
    )
    pixels = np.array([[[0.2, 0.4, 0.8, 0.7]]], dtype=np.float32)
    actual = ocio_bridge.apply_file_transform(pixels, path, "forward")
    expected = pixels.copy()
    expected[..., :3] *= 0.5
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    restored = ocio_bridge.apply_file_transform(actual, path, "inverse")
    np.testing.assert_allclose(restored, pixels, atol=1e-6, rtol=0)


def test_bad_inputs_propagate_errors(tmp_path, config):
    with pytest.raises(ValueError, match="direction"):
        ocio_bridge.apply_file_transform(np.ones(3), "missing.cube", "sideways")
    with pytest.raises(Exception):
        ocio_bridge.apply_file_transform(np.ones(3), tmp_path / "missing.cube")
    with pytest.raises(Exception):
        ocio_bridge.get_config(tmp_path / "missing.ocio")
    with pytest.raises(Exception):
        ocio_bridge.apply_transform(np.ones(3), config, "unknown", "destination")
    with pytest.raises(ValueError, match="RGB/RGBA"):
        ocio_bridge.apply_transform(np.ones((2, 2)), config, "source", "destination")
