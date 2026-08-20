from pydantic import BaseModel, Field, field_validator


class DatasetWeightsConfig(BaseModel):
    __root__: dict[str, float] = Field(default_factory=dict)

    @field_validator('__root__')
    def validate_weights(cls, v):
        for name, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for {name} must be positive")
        return v
