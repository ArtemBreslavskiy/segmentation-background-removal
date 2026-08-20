from pydantic import BaseModel, Field


class SplitsConfig(BaseModel):
    seed: int
    val_ratio: float = Field(ge=0)
    test_ratio: float = Field(ge=0)
