from pydantic import BaseModel, Field
from src.configs.schemas.dataset.splits import SplitsConfig
from src.configs.schemas.dataset.resizes import ResizesConfig
from src.configs.schemas.dataset.weights import DatasetWeightsConfig


class BaseDatasetConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str


class BinarySegmentationDataset(BaseDatasetConfig):
    max_area: int = Field(gt=0)
    splits: SplitsConfig
    resizes: ResizesConfig
    weights: DatasetWeightsConfig
