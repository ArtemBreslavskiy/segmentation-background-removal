from src.configs.schemas.model.model_init import BaseModelInitConfig


def create_model(config: BaseModelInitConfig):
    if config.type == "deeplab_v3_plus":
        from src.models.DeepLabV3Plus import DeepLabV3Plus
        model_params = config.model_dump(exclude={"type"})
        return DeepLabV3Plus(**model_params)

    elif config.type == "seg_former":
        from src.models.SegFormer import SegFormer
        model_params = config.model_dump(exclude={"type"})
        return SegFormer(**model_params)

    else:
        raise ValueError(f"Unknown model type: {config.type}")
