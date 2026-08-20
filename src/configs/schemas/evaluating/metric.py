from pydantic import BaseModel
from typing import Literal


class BaseMetric(BaseModel):
    model_config = {"extra": "forbid"}
    name: str


class IOUConfig(BaseModel):
    name: Literal["iou"]
    task: str


class AccuracyConfig(BaseModel):
    name: Literal["accuracy"]
    task: str


class PrecisionConfig(BaseModel):
    name: Literal["precision"]
    task: str


class RecallConfig(BaseModel):
    name: Literal["recall"]
    task: str


class F1Config(BaseModel):
    name: Literal["f1"]
    task: str


class SpecificityConfig(BaseModel):
    name: Literal["specificity"]
    task: str


class MCCConfig(BaseModel):
    name: Literal["mcc"]
    task: str

