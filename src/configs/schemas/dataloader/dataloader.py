from pydantic import BaseModel, Field
from src.configs.schemas.dataloader.batch_sizes import BatchSizesConfig
from src.configs.schemas.dataloader.shuffle import ShuffleConfig
from src.configs.schemas.dataloader.samplers import SamplersConfig
from src.configs.schemas.dataloader.pad_collate import PadCollateConfig


class DataloaderConfig(BaseModel):
    seed: int
    num_workers: int = Field(2, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = Field(2, ge=0)
    batch_sizes: BatchSizesConfig
    shuffle: ShuffleConfig
    samplers: SamplersConfig
    pad_collate: PadCollateConfig
