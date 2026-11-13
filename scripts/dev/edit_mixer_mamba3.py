"""One-off edit script used to author patches/mamba/0004 (Mamba3 as a MixerModel layer)."""

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

p = "mamba_ssm/models/mixer_seq_simple.py"
s = open(p).read()
s = sub(s, "from mamba_ssm.modules.mamba2 import Mamba2\n",
        "from mamba_ssm.modules.mamba2 import Mamba2\nfrom mamba_ssm.modules.mamba3 import Mamba3\n")
s = sub(s, """        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,""", """        layer_classes = {"Mamba1": Mamba, "Mamba2": Mamba2, "Mamba3": Mamba3}
        if ssm_layer not in layer_classes:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1, Mamba2 and Mamba3")
        mixer_cls = partial(
            layer_classes[ssm_layer],""")
open(p, "w").write(s)
print("mixer edited")
