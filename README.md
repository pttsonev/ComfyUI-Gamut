# ComfyUI-Gamut

Colour management and tagged EXR output for ComfyUI, including the LTX HDR
IC-LoRA path. Images stay as ordinary float32 `IMAGE` tensors; a separate
`COLORSPACE` connection carries primaries, transfer curve, and provenance.

## Install

Clone this repository into `ComfyUI/custom_nodes/ComfyUI-Gamut`, then install
with **the Python interpreter used by ComfyUI**:

```sh
cd ComfyUI/custom_nodes/ComfyUI-Gamut
python -m pip install .
```

Restart ComfyUI. Seven core nodes appear under `Gamut`. Python 3.10 or newer,
torch, NumPy, and OpenEXR's modern `File` bindings (3.3 or newer) are required.
OpenEXR 3.4.4 is the verified IO version. PNG/TIFF writing needs no additional
imaging library.

For the three optional OCIO nodes:

```sh
python -m pip install ".[ocio]"
```

With `opencolorio` absent, the core nodes still load and a single informational
log message explains why OCIO nodes were omitted. Imports also work outside
ComfyUI; relative output prefixes require ComfyUI's path service, while absolute
prefixes work in standalone use.

## Nodes

| Display name | Purpose | Outputs |
|---|---|---|
| Gamut: Color Space Convert | Rotate linear RGB between Rec.709, ACEScg, AP0, and Rec.2020 | IMAGE, COLORSPACE |
| Gamut: Transfer | Decode to linear light or encode a transfer curve | IMAGE, COLORSPACE |
| Gamut: Load EXR | Read float RGB/RGBA and real header tags | IMAGE, MASK, COLORSPACE |
| Gamut: Save EXR | Write tagged half/float EXR frames without overwriting | Output node |
| Gamut: Save High Bit | Write 16-bit PNG/TIFF with optional dither | Output node |
| Gamut: LTX HDR Decode | Decode LogC3 HDR IC-LoRA VAE output | IMAGE, COLORSPACE |
| Gamut: OCIO Transform | Convert between spaces in an OCIO config | IMAGE, COLORSPACE |
| Gamut: OCIO Display View | Apply a display/view from an explicit source space | IMAGE |
| Gamut: Apply LUT | Apply a file transform with direction and interpolation | IMAGE |
| Gamut: Color Space Info | Report a declaration, tensor layout, or EXR header | STRING |

OCIO Transform, OCIO Display View, and Apply LUT register only when OCIO is
available. All nodes accept an optional `COLORSPACE` socket. Transforming nodes
emit a new declaration; terminal writers, display/LUT outputs, and the info
node use the output contracts in the table.

## Colour state and transfer rules

`COLORSPACE` is a ComfyUI custom socket type whose payload is the immutable
`ColorSpace(primaries, transfer, source)` dataclass. Matching socket type names
provide the registration and connection contract. The package also exports
`CUSTOM_TYPES["COLORSPACE"]` for Python consumers.

- Primaries: `rec709`, `acescg` (AP1), `ap0`, `rec2020`.
- Transfers: `linear`, `srgb`, `logc3` (EI800), `acescct`, `g22`, `g24`.
- Provenance includes `exr:chromaticities`, `exr:colorSpace`, `widget`,
  `widget:fallback`, `node:<name>`, and `ocio:<space>`.

Wire `COLORSPACE` alongside `IMAGE`. It overrides source widgets. Load EXR gives
real header metadata priority; for untagged or unrecognised metadata it uses a
wired declaration or its fallback widgets, recording `widget:fallback` for the
latter. Unknown chromaticities remain visible verbatim in Color Space Info.

**Untagged input to Color Space Convert is assumed linear light.** Tagged
non-linear input raises an error directing you to `Gamut: Transfer` (decode).
Transfer's decode direction uses the incoming declaration's curve; its encode
direction requires linear input and uses the selected output curve. With no
declaration, its primaries widget supplies the output primaries declaration.
RGB transfer operations preserve alpha. Gamma is sign-preserving, and generic
curves retain negative values and highlights by default.

Save EXR's primaries/transfer widgets declare the existing pixels; saving does
not perform a colour conversion. Select a conversion node first when the
pixel values need to change.

## LTX HDR to ACEScg EXR

For **encoded VAE output** from the LogC3 HDR IC-LoRA:

1. Connect the VAE `IMAGE` to `Gamut: LTX HDR Decode`.
2. Select `acescg` and the desired exposure in stops.
3. Connect both `IMAGE` and `COLORSPACE` to `Gamut: Save EXR`.
4. Choose a prefix and version, then queue the workflow.

The adapter clamps only its VAE input to `[0,1]`, decodes ordinary LogC3 codes
without a `*2-1` rescale, and preserves output values above 1. A code of `0.5`
decodes to approximately `0.513383`; a code of `1.0` produces approximately
`55.079577` before exposure. Negative output is clipped by this adapter's default.

If an upstream LTX node **already outputs linear Rec.709**, connect it directly
to Color Space Convert (`rec709` to `acescg`) and then Save EXR. Do not decode
linear output a second time.

## EXR loading and saving

Load EXR returns RGB as `[B,H,W,3]` and a same-size `MASK` containing transparency
(`1-alpha`), or zeros for RGB-only files. Save EXR accepts RGB or RGBA batches;
an RGBA input's fourth channel is alpha, not a ComfyUI transparency mask.

Leave `tonemap_preview` **off** for HDR work. Enabling it produces a display
preview by decoding, rotating to Rec.709, clipping to `[0,1]`, and encoding sRGB.
The returned declaration changes to Rec.709/sRGB. This uses the existing colour
curves, not a custom tone-mapping operator.

Relative save prefixes use `folder_paths.get_save_image_path()` and ComfyUI's
output directory. Absolute prefixes are honoured directly. Use a prefix without
an extension. For example, prefix `shots/beauty`, version `1`, start frame `1001`,
and padding `4` produce:

```text
shots/beauty_v001.1001.exr
shots/beauty_v001.1002.exr
```

Every destination is checked before the batch starts, and exclusive creation
also refuses collisions during saving. Existing files are never overwritten.
Increase the version or change the prefix for another render. ZIP and half float
are the defaults; float32 and other supported lossless compressions are selectable.

EXR headers carry eight `chromaticities` floats and a `colorSpace` string. The
canonical name map in `gamut.exr` supplies complete names where defined:

| Primaries / transfer | Written `colorSpace` |
|---|---|
| ACEScg / linear | ACEScg |
| AP0 / linear | ACES2065-1 |
| Rec.709 / linear | Linear Rec.709 (sRGB) |
| Rec.2020 / linear | Linear Rec.2020 |
| Rec.709 / sRGB | sRGB |
| ACEScg / ACEScct | ACEScct |
| Rec.709 / LogC3 | LogC3 |

Other combinations retain their chromaticities and use the transfer display
name as fallback. No colour is inferred from the appearance of the pixels.

## High-bit export and OCIO

Save High Bit treats untagged input as linear. A wired transfer declaration is
decoded before applying the selected output transfer when they differ. Primaries
are preserved. Output is rounded to 16-bit integer codes and clipped to `[0,1]`;
use EXR when negative values or HDR highlights must survive. TPDF dither is
optional, off by default, and applied before quantisation. These files use the
same version/frame naming discipline, beginning at `_v001.1001`.

The default OCIO config is `cg-config-v4.0.0_aces-v2.0_ocio-v2.5`; the config field
also accepts a built-in name, `$OCIO`, or a config file path. Dropdowns list the
default config's spaces/displays/views. For custom config names, convert those
widgets to inputs and connect strings; the selected config validates the names
at execution time.

OCIO Transform and Display View resolve a wired declaration through the shared
canonical name map or its `ocio:<space>` provenance. An unmatched declaration
raises an error. Display View always takes an explicit source; it never uses
the config's `scene_linear` role as a guess. Arbitrary OCIO output spaces retain
their name in provenance and use `unknown` primaries/transfer until transformed
back to a recognised space. Display and LUT operations return IMAGE only because
their results cannot in general be described by the core declaration registry.

## Development checks

Use the repository's Python environment and run the **full** suite:

```sh
python -m pip install ".[test]"
python -m pytest -q -ra
python -m compileall -q gamut nodes tests __init__.py
python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir --wheel-dir dist
```

The tests derive all primaries matrices independently using Bradford adaptation.
OCIO integration checks skip when its optional binding is absent. CUDA tests use
the real kernel probe in `tests.conftest.DEVICES` and `requires_cuda`; driver
visibility alone is insufficient. No separate lint/type-check tool is configured.

MIT licensed, copyright 2026 Petar Tsonev. See `LICENSE` and `NOTICE` for the
spacepxl attribution covering IO discipline and the piecewise sRGB formulation.
