import contextlib
import hashlib
import os

# cuBLAS reads this at CUDA initialization; it must be set before torch touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


@contextlib.contextmanager
def deterministic(enabled: bool):
    """Toggle exactly the way a user would: torch.use_deterministic_algorithms."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def digest(tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"none")
            continue
        t = t.detach()
        if t.is_complex():
            t = torch.view_as_real(t)
        h.update(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def n_unique(fn, runs=6) -> int:
    """How many distinct (bitwise) results `fn` produces over identical runs."""
    return len({digest(fn()) for _ in range(runs)})


def cc():
    return torch.cuda.get_device_capability()


def _triton_version():
    try:
        import triton
        from packaging.version import Version
        return Version(triton.__version__)
    except ImportError:
        return None


# Mamba3's SISO forward kernel contains a tl.dot with N = 1, which Triton <= 3.4 rejects
# ("M >= 16, N >= 16 and K >= 16") on every GPU. Upstream mamba_ssm requires triton >= 3.5.0.
_tv = _triton_version()
requires_mamba3_triton = pytest.mark.skipif(
    _tv is None or _tv.release < (3, 5),
    reason=f"Mamba3 Triton kernels need triton >= 3.5.0 (found {_tv}); torch 2.8 pins triton 3.4",
)
