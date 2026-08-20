from pydantic import BaseModel


class ShuffleConfig(BaseModel):
    train: bool | None
    test: bool | None
    val: bool | None
