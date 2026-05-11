import os


def apply_runtime_env(config: dict):
    rt = config.get("runtime", {})

    # --- CUDA allocator ---
    alloc_opts = []
    if rt.get("cuda_alloc", {}).get("expandable_segments", False):
        alloc_opts.append("expandable_segments:True")
    max_split = rt.get("cuda_alloc", {}).get("max_split_size_mb", 0)
    if max_split > 0:
        alloc_opts.append(f"max_split_size_mb:{max_split}")
    if alloc_opts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_opts)

    # --- NVML check ---
    if rt.get("nvml_check", False):
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    # --- cuDNN cache ---
    cache_limit = rt.get("cudnn_cache_limit", None)
    if cache_limit is not None:
        os.environ["TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT"] = str(cache_limit)

    # --- TF32 ---
    if rt.get("allow_tf32", True):
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

    # --- Torch Compile / Inductor settings ---
    tc = rt.get("torch_compile", {})
    if tc.get("disable_triton", False):
        os.environ["TORCH_COMPILE_DISABLE_TRITON"] = "1"
    if tc.get("cpp_wrapper", False):
        os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "1"
    if not tc.get("max_autotune", True):
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
