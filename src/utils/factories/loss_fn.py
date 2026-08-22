import torch.nn as nn
from src.configs.schemas.learning.loss import BaseLossConfig


def create_loss(config: BaseLossConfig) -> nn.Module:
    if config.type == "focal":
        from src.losses.FocalLoss import FocalLoss
        loss_params = config.model_dump(exclude={"type"})
        return FocalLoss(**loss_params)

    elif config.type == "soft_dice":
        from src.losses.SoftDiceLoss import SoftDiceLoss
        loss_params = config.model_dump(exclude={"type"})
        return SoftDiceLoss(**loss_params)

    elif config.type == "masked_focal":
        from src.losses.MaskedFocalLoss import MaskedFocalLoss
        loss_params = config.model_dump(exclude={"type"})
        return MaskedFocalLoss(**loss_params)

    elif config.type == "masked_soft_dice":
        from src.losses.MaskedSoftDiceLoss import MaskedSoftDiceLoss
        loss_params = config.model_dump(exclude={"type"})
        return MaskedSoftDiceLoss(**loss_params)

    elif config.type == "masked_tversky":
        from src.losses.MaskedTverskyLoss import MaskedTverskyLoss
        loss_params = config.model_dump(exclude={"type"})
        return MaskedTverskyLoss(**loss_params)

    elif config.type == "combo":
        from src.losses.ComboLoss import ComboLoss
        losses = []
        for loss_conf in config.loss_functions:
            losses.append(create_loss(loss_conf))
        loss_params = config.model_dump(exclude={"type", "loss_functions"})
        return ComboLoss(loss_functions=losses, **loss_params)

    else:
        raise ValueError(f"Unknown loss type: {config.type}")
