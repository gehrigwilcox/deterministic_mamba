"""Bitwise-deterministic fork of the mamba_ssm 2.2.4 Mamba1 kernels and MixerModel.

Drop-in usage (same constructor args and state_dict keys as mamba_ssm 2.2.4):

    from deterministic_mamba import MixerModel, MambaLMHeadModel, Mamba
"""

__version__ = "2.2.4.post1"

import os
import warnings

import torch

from deterministic_mamba import _C  # noqa: F401  (compiled extension)
from deterministic_mamba.config import set_workspace_mb, workspace_mb
from deterministic_mamba.ops.causal_conv1d import causal_conv1d_fn
from deterministic_mamba.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from deterministic_mamba.modules.mamba_simple import Mamba
from deterministic_mamba.models.config_mamba import MambaConfig
from deterministic_mamba.models.mixer_seq_simple import MambaLMHeadModel, MixerModel, create_block


def enable_full_determinism(seed=None, warn_only=False):
    """Configure PyTorch so that the *rest* of the training step is deterministic too.

    The Mamba kernels in this package are deterministic on their own. Everything around them
    (cuBLAS GEMMs in the projections, embedding backward, cuDNN, your loss, dataloader...) is
    governed by PyTorch's global settings, which this helper sets:

    * ``torch.use_deterministic_algorithms(True)``
    * ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (required by cuBLAS for reproducible GEMMs; must be set
      before the first cuBLAS call, so call this function early, or export it in the shell)
    * cuDNN deterministic / no benchmark
    * TF32 off (TF32 is deterministic, but toggling it between runs changes results)
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        if torch.cuda.is_initialized():
            warnings.warn(
                "CUBLAS_WORKSPACE_CONFIG was not set before CUDA initialization; export "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment for reproducible cuBLAS GEMMs."
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if seed is not None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "Mamba",
    "MambaConfig",
    "MambaLMHeadModel",
    "MixerModel",
    "create_block",
    "selective_scan_fn",
    "mamba_inner_fn",
    "causal_conv1d_fn",
    "enable_full_determinism",
    "set_workspace_mb",
    "workspace_mb",
]
