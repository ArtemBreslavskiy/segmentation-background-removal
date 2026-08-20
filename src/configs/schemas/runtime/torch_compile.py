from pydantic import BaseModel, Field


class TorchCompileConfig(BaseModel):
    disable_triton: bool = False
    cpp_wrapper: bool = False
    max_autotune: bool = False
