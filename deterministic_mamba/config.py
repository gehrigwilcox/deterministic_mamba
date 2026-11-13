"""Runtime configuration for deterministic_mamba."""

import os

_DEFAULT_WORKSPACE_MB = 256
_workspace_mb = None


def workspace_mb() -> int:
    """Memory budget (MiB) for the per-channel dB/dC workspace of the selective-scan backward.

    The deterministic backward writes each channel's dB/dC contribution to its own buffer and
    reduces them in order, instead of atomically accumulating. Channels are processed in slices
    sized to fit this budget: a larger budget means fewer kernel launches (faster), a smaller one
    means less peak memory. The result is bitwise identical for any budget.

    Set with ``set_workspace_mb`` or the ``DETERMINISTIC_MAMBA_WORKSPACE_MB`` env var.
    """
    if _workspace_mb is not None:
        return _workspace_mb
    return int(os.environ.get("DETERMINISTIC_MAMBA_WORKSPACE_MB", _DEFAULT_WORKSPACE_MB))


def set_workspace_mb(mb: int) -> None:
    global _workspace_mb
    if mb < 1:
        raise ValueError("workspace budget must be >= 1 MiB")
    _workspace_mb = int(mb)
