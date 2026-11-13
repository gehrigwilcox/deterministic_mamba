"""End-to-end determinism of a MixerModel training loop (the user-facing guarantee)."""

import hashlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch

DEVICE = "cuda"


def _train(steps=4, dtype=torch.float32, use_fast_path=True, seed=0, seqlen=300, return_model=False):
    from deterministic_mamba import MixerModel

    torch.manual_seed(seed)
    model = MixerModel(
        d_model=128,
        n_layer=3,
        d_intermediate=0,
        vocab_size=97,
        ssm_cfg={"use_fast_path": use_fast_path},
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=DEVICE,
    )
    head = torch.nn.Linear(128, 97, bias=False, device=DEVICE)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(1234)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, 97, (4, seqlen + 1), generator=g).to(DEVICE)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            h = model(ids[:, :-1])
            logits = head(h).float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 97), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if return_model:
        return model, head, losses
    return {n: p.detach().clone() for n, p in model.named_parameters()}, losses


def _digest(state):
    h = hashlib.sha256()
    for n in sorted(state):
        h.update(n.encode())
        h.update(state[n].float().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_training_is_bitwise_reproducible_in_process(dtype):
    a, la = _train(dtype=dtype)
    b, lb = _train(dtype=dtype)
    assert la == lb
    for n in a:
        assert torch.equal(a[n], b[n]), f"parameter {n} diverged between identical runs"


def test_training_is_bitwise_reproducible_across_processes():
    script = textwrap.dedent(
        f"""
        import sys, torch
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from deterministic_mamba import enable_full_determinism
        enable_full_determinism()
        from test_mixer_model import _train, _digest
        state, losses = _train()
        print(_digest(state))
        """
    )
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    digests = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
        .stdout.strip().splitlines()[-1]
        for _ in range(2)
    ]
    assert digests[0] == digests[1]


def test_fast_and_slow_paths_agree():
    fast, lf = _train(steps=2, use_fast_path=True)
    slow, ls = _train(steps=2, use_fast_path=False)
    for a, b in zip(lf, ls):
        assert abs(a - b) < 1e-4
    for n in fast:
        torch.testing.assert_close(fast[n], slow[n], rtol=1e-3, atol=1e-4)


def test_works_under_torch_deterministic_mode():
    from deterministic_mamba import enable_full_determinism

    prev = torch.are_deterministic_algorithms_enabled()
    try:
        enable_full_determinism()
        _train(steps=1)
    finally:
        torch.use_deterministic_algorithms(prev)


def test_lm_head_model_and_step_decoding():
    from deterministic_mamba import MambaConfig, MambaLMHeadModel

    torch.manual_seed(0)
    cfg = MambaConfig(d_model=64, n_layer=2, vocab_size=50, rms_norm=True, fused_add_norm=True,
                      residual_in_fp32=True, pad_vocab_size_multiple=8)
    model = MambaLMHeadModel(cfg, device=DEVICE).eval()
    ids = torch.randint(0, 50, (2, 12), device=DEVICE)
    with torch.no_grad():
        full = model(ids).logits
        # Recurrent decoding must reproduce the parallel scan.
        from deterministic_mamba.utils.generation import InferenceParams

        ip = InferenceParams(max_seqlen=12, max_batch_size=2)
        step_logits = []
        for t in range(12):
            out = model(ids[:, t : t + 1], inference_params=ip).logits
            ip.seqlen_offset += 1
            step_logits.append(out)
    torch.testing.assert_close(torch.cat(step_logits, 1), full, rtol=1e-3, atol=1e-3)
