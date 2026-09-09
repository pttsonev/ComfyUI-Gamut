"""Shared test fixtures.

The device list is probed, not merely queried. ``torch.cuda.is_available()``
returns True whenever a CUDA device is visible to the driver, even when the
installed torch build ships no kernel image for that device's compute
capability — an RTX 5090 (sm_120) under a cu124 wheel reports available and
then raises "no kernel image is available for execution on the device" on the
first real op. Running an actual kernel is the only honest test.
"""

import pytest
import torch


def _cuda_really_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # A real kernel launch plus a sync — allocation alone does not fault.
        probe = torch.ones(8, device="cuda") * 2.0
        torch.cuda.synchronize()
        return bool(probe.sum().item() == 16.0)
    except Exception:
        return False


CUDA_OK = _cuda_really_works()
DEVICES = ["cpu"] + (["cuda"] if CUDA_OK else [])

requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="no usable CUDA device")


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param
