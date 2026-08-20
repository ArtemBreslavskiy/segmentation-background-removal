from pydantic import BaseModel, Field


class CudaAllocConfig(BaseModel):
    expandable_segments: bool = True
    max_split_size_mb: int = Field(0, ge=0)
