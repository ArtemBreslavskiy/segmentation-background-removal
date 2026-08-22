from pydantic import BaseModel, Field
from src.configs.schemas.dataset.splits import SplitsConfig
from src.configs.schemas.dataset.resizes import ResizesConfig


class DatasetConfig(BaseModel):
    max_area: int = Field(gt=0)
    splits: SplitsConfig
    resizes: ResizesConfig
