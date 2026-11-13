"""End-to-end: MixerModel training with each Mamba version is bitwise reproducible once
torch.use_deterministic_algorithms(True) is on - including when it's turned on after import."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from conftest import deterministic, digest, requires_mamba3_triton

DEV = "cuda"
LAYERS = ["Mamba1", "Mamba2", pytest.param("Mamba3", marks=requires_mamba3_triton)]
LAYER_CFG = {
    "Mamba1": {"layer": "Mamba1"},
    "Mamba2": {"layer": "Mamba2", "d_state": 64, "headdim": 32, "chunk_size": 64},
    "Mamba3": {"layer": "Mamba3", "d_state": 64, "headdim": 32, "chunk_size": 64},
}


def train(layer, steps=3, seqlen=200, seed=0):
    from mamba_ssm.models.mixer_seq_simple import MixerModel

    torch.manual_seed(seed)
    model = MixerModel(d_model=128, n_layer=2, d_intermediate=0, vocab_size=97, ssm_cfg=dict(LAYER_CFG[layer]),
                       rms_norm=True, fused_add_norm=True, residual_in_fp32=True, device=DEV)
    head = torch.nn.Linear(128, 97, bias=False, device=DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        ids = torch.randint(0, 97, (4, seqlen + 1), generator=g).to(DEV)
        loss = torch.nn.functional.cross_entropy(head(model(ids[:, :-1])).float().reshape(-1, 97),
                                                 ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return [p.detach() for p in model.parameters()]


@pytest.mark.parametrize("layer", LAYERS)
def test_training_bitwise_reproducible_in_process(layer):
    with deterministic(True):
        assert digest(train(layer)) == digest(train(layer))


@pytest.mark.parametrize("layer", LAYERS)
def test_training_bitwise_reproducible_across_processes_toggled_after_import(layer):
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {os.path.dirname(__file__)!r})
        import torch
        import mamba_ssm.models.mixer_seq_simple       # import first ...
        torch.use_deterministic_algorithms(True)       # ... then turn determinism on
        from test_mixer_model import train
        from conftest import digest
        print(digest(train({layer!r})))
    """)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    env.pop("MAMBA_DETERMINISTIC", None)
    out = [subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
           for _ in range(2)]
    for o in out:
        assert o.returncode == 0, o.stderr[-3000:]
    assert out[0].stdout.split()[-1] == out[1].stdout.split()[-1]
