import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
CSRC = "csrc"


def cuda_archs():
    # Respect TORCH_CUDA_ARCH_LIST if the user set it; otherwise build for common archs.
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    flags = []
    for arch in ["75", "80", "86", "89", "90"]:
        flags += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    return flags


sources = [
    f"{CSRC}/bindings.cpp",
    f"{CSRC}/det_reduce.cu",
    f"{CSRC}/selective_scan/selective_scan.cpp",
    *[f"{CSRC}/selective_scan/selective_scan_fwd_{t}.cu" for t in ("fp32", "fp16", "bf16")],
    *[f"{CSRC}/selective_scan/selective_scan_bwd_{t}.cu" for t in ("fp32", "fp16", "bf16")],
    f"{CSRC}/causal_conv1d/causal_conv1d.cpp",
    f"{CSRC}/causal_conv1d/causal_conv1d_fwd.cu",
    f"{CSRC}/causal_conv1d/causal_conv1d_bwd.cu",
]

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo",
] + cuda_archs()

setup(
    name="deterministic_mamba",
    version="2.2.4.post1",
    description="Bitwise-deterministic fork of the mamba_ssm 2.2.4 Mamba1 kernels and MixerModel",
    packages=find_packages(include=["deterministic_mamba", "deterministic_mamba.*"]),
    ext_modules=[
        CUDAExtension(
            name="deterministic_mamba._C",
            sources=sources,
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
            include_dirs=[str(ROOT / CSRC)],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "einops", "triton", "packaging"],
)
