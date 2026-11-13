"""One-off edit script used to author patches/mamba/0001 (Python side). Run from an upstream mamba checkout."""

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

p = "mamba_ssm/utils/determinism.py"
s = open(p).read()
s = sub(s, """def set_deterministic_mode(value):
    global _deterministic_override
    _deterministic_override = value
""", """def set_deterministic_mode(value):
    global _deterministic_override
    _deterministic_override = value


def selective_scan_workspace_mb():
    \"\"\"Memory budget (MiB) for the per-channel dB/dC workspace of the deterministic Mamba1
    selective-scan backward (MAMBA_DETERMINISTIC_WORKSPACE_MB, default 256). Larger is faster;
    the result is bitwise identical for any value.\"\"\"
    return int(os.environ.get("MAMBA_DETERMINISTIC_WORKSPACE_MB", "256"))
""")
open(p, "w").write(s)

p = "mamba_ssm/ops/selective_scan_interface.py"
s = open(p).read()
s = sub(s, "import selective_scan_cuda\n", """import selective_scan_cuda

from mamba_ssm.utils.determinism import use_deterministic_mode, selective_scan_workspace_mb
""")
s = sub(s, """            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )""", """            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False,  # option to recompute out_z, not used here
            deterministic=use_deterministic_mode(),
            workspace_mb=selective_scan_workspace_mb(),
        )""")
s = sub(s, """            ctx.delta_softplus,
            True  # option to recompute out_z
        )""", """            ctx.delta_softplus,
            True,  # option to recompute out_z
            deterministic=use_deterministic_mode(),
            workspace_mb=selective_scan_workspace_mb(),
        )""")
open(p, "w").write(s)
print("python edited")
