from pydantic import BaseModel
from src.configs.schemas.runtime.cuda_alloc import CudaAllocConfig
from src.configs.schemas.runtime.torch_compile import TorchCompileConfig


class RuntimeConfig(BaseModel):
    cuda_alloc: CudaAllocConfig
    nvml_check: bool
    cudnn_cache_limit: int
    allow_tf32: bool
    torch_compile: TorchCompileConfig
