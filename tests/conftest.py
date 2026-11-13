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
