import os

# cuBLAS reads this at CUDA initialization; it must be set before torch touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
