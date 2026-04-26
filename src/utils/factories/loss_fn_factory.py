from typing import Dict

from src.utils.factories.factory_utils import convert_value, get_class


def create_loss(config: Dict):
    if config["learning"]["loss"]["class"].endswith("ComboLoss"):
        from src.losses.ComboLoss import ComboLoss

        losses = []
        for loss_item in config["learning"]["loss"]["params"]["loss_functions"]:
            ClassLoss = get_class(loss_item["class"])
            losses.append(ClassLoss(**loss_item["params"]))
        return ComboLoss(
            loss_functions=losses,
            weights=convert_value(config["learning"]["loss"]["params"]["weights"]),
        )
    else:
        LossClass = get_class(config["learning"]["loss"]["class"])
        return LossClass(**convert_value(config["learning"]["loss"]["params"]))
