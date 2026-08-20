from pydantic import BaseModel, Field


class BatchSizesConfig(BaseModel):
    train: int | None = Field(gt=0)
    test: int | None = Field(gt=0)
    val: int | None = Field(gt=0)
