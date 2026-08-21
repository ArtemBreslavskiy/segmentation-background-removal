import os
from src.configs.schemas.runtime.runtime import RuntimeConfig


def apply_runtime_env(config: RuntimeConfig):
    # --- CUDA allocator ---
    alloc_opts = []
    if config.cuda_alloc.expandable_segments:
        alloc_opts.append("expandable_segments:True")
    max_split = config.cuda_alloc.max_split_size_mb
    if max_split > 0:
        alloc_opts.append(f"max_split_size_mb:{max_split}")
    if alloc_opts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_opts)

    # --- NVML check ---
    if config.nvml_check:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    # --- cuDNN cache ---
    cache_limit = config.cudnn_cache_limit
    if cache_limit is not None:
        os.environ["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = str(cache_limit)

    # --- TF32 ---
    if config.allow_tf32:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

    # --- Torch Compile / Inductor settings ---
    if config.torch_compile.disable_triton:
        os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
    if config.torch_compile.cpp_wrapper:
        os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "1"
    if not config.torch_compile.max_autotune:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
