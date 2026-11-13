"""One-off edit script used to author patches/mamba/0003 (Mamba3 SISO deterministic dK state post-step).
Run from an upstream mamba checkout (after 0002)."""

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

p = "mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py"
s = open(p).read()
s = sub(s, "from mamba_ssm.utils.determinism import autotune\n",
        "from mamba_ssm.utils.determinism import autotune, use_deterministic_mode\n")
# kernel signature: workspace outputs + DETERMINISTIC flag
s = sub(s, """    # Outputs tensors
    dK, dK_bias, dAngles,
    # Strides for dK_State: (num_sequences, nheads, headdim_qk)""", """    # Outputs tensors
    dK, dK_bias, dAngles,
    # Deterministic-mode workspaces: (num_sequences, nheads, HEADDIM_QK) and (num_sequences, nheads, HEADDIM_QK // 2)
    WS_dK, WS_dAngles,
    # Strides for dK_State: (num_sequences, nheads, headdim_qk)""")
s = sub(s, """    GQA_RATIO: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    \"\"\"
    Post-kernel for d_ok_state contributions.""", """    GQA_RATIO: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    \"\"\"
    Post-kernel for d_ok_state contributions.
    In DETERMINISTIC mode nothing is accumulated here: each program stores its dk_rotated and
    dtheta_k into WS_dK / WS_dAngles and the host reduces them in a fixed order.""")
s = sub(s, """    # 1. Accumulate to dK (GQA reduction via atomic)
    dk_base = dK + pid_batch * stride_dk_batch + last_pos * stride_dk_seqlen + qk_head_idx * stride_dk_head
    tl.atomic_add(dk_base + offs_d * stride_dk_qkdim, dk_rotated, mask=offs_d < headdim_qk)
    
    # 2. Accumulate to dK_bias (batch reduction via atomic)
    dk_bias_base = dK_bias + pid_head * stride_dk_bias_head
    tl.atomic_add(dk_bias_base + offs_d * stride_dk_bias_qkdim, dk_rotated, mask=offs_d < headdim_qk)
""", """    ws_row = (pid_batch + pid_seq) * tl.num_programs(0) + pid_head
    if DETERMINISTIC:
        tl.store(WS_dK + ws_row * HEADDIM_QK + offs_d, dk_rotated)
    else:
        # 1. Accumulate to dK (GQA reduction via atomic)
        dk_base = dK + pid_batch * stride_dk_batch + last_pos * stride_dk_seqlen + qk_head_idx * stride_dk_head
        tl.atomic_add(dk_base + offs_d * stride_dk_qkdim, dk_rotated, mask=offs_d < headdim_qk)

        # 2. Accumulate to dK_bias (batch reduction via atomic)
        dk_bias_base = dK_bias + pid_head * stride_dk_bias_head
        tl.atomic_add(dk_bias_base + offs_d * stride_dk_bias_qkdim, dk_rotated, mask=offs_d < headdim_qk)
""")
s = sub(s, """    da_base = dAngles + pid_batch * stride_dangles_batch + last_pos * stride_dangles_seqlen + pid_head * stride_dangles_head
    tl.atomic_add(da_base + offs_dr * stride_dangles_qkdim, dtheta_k, mask=offs_dr < headdim_angles)
""", """    if DETERMINISTIC:
        tl.store(WS_dAngles + ws_row * (HEADDIM_QK // 2) + offs_dr, dtheta_k)
    else:
        da_base = dAngles + pid_batch * stride_dangles_batch + last_pos * stride_dangles_seqlen + pid_head * stride_dangles_head
        tl.atomic_add(da_base + offs_dr * stride_dangles_qkdim, dtheta_k, mask=offs_dr < headdim_angles)
""")
# host
s = sub(s, """    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    
    grid = (nheads, num_sequences)
    
    mamba3_siso_bwd_kernel_dk_state_post[grid](
        # Input tensors
        d_ok_state, angles, k, k_bias, Cu_Seqlens,
        # Output tensors
        dk, dk_bias, dangles,
""", """    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    
    grid = (nheads, num_sequences)

    deterministic = use_deterministic_mode()
    if deterministic:
        ws_dk = torch.empty((num_sequences, nheads, HEADDIM_QK), dtype=torch.float32, device=dk.device)
        ws_dangles = torch.empty((num_sequences, nheads, HEADDIM_QK // 2), dtype=torch.float32, device=dk.device)
    else:
        ws_dk = ws_dangles = None
    
    mamba3_siso_bwd_kernel_dk_state_post[grid](
        # Input tensors
        d_ok_state, angles, k, k_bias, Cu_Seqlens,
        # Output tensors
        dk, dk_bias, dangles,
        ws_dk, ws_dangles,
""")
s = sub(s, """        GQA_RATIO=GQA_RATIO,
        IS_VARLEN=is_varlen,
        num_warps=2,
        num_stages=3,
    )
""", """        GQA_RATIO=GQA_RATIO,
        IS_VARLEN=is_varlen,
        DETERMINISTIC=deterministic,
        num_warps=2,
        num_stages=3,
    )

    if deterministic:
        # Fixed-order reductions replacing the kernel's atomics.
        if is_varlen:
            last_pos = (Cu_Seqlens[1:] - 1).long()
            batch_idx = torch.zeros_like(last_pos)
        else:
            last_pos = torch.full((num_sequences,), seqlen - 1, dtype=torch.long, device=dk.device)
            batch_idx = torch.arange(num_sequences, dtype=torch.long, device=dk.device)
        dk_rot = ws_dk[..., :headdim_qk]                                  # (seq, nheads, d)
        # dK: GQA reduction (heads h*GQA .. h*GQA+GQA-1 share qk head h)
        dk_gqa = dk_rot.view(num_sequences, nheads_qk, GQA_RATIO, headdim_qk).sum(dim=2)
        dk.index_put_((batch_idx, last_pos), dk_gqa.to(dk.dtype), accumulate=True)
        # dK_bias: reduction over sequences
        dk_bias.add_(dk_rot.sum(dim=0).to(dk_bias.dtype))
        # dAngles: one contribution per (sequence, head)
        dangles[..., :headdim_angles].index_put_(
            (batch_idx, last_pos), ws_dangles[..., :headdim_angles].to(dangles.dtype), accumulate=True)
""")
open(p, "w").write(s)
print("mamba3 edited")
