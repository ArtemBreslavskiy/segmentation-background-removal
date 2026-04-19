import importlib
from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim


def _get_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _convert_value(value: Any):
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)

            float_val = float(value)
            if float_val.is_integer() and "." not in value:
                return int(float_val)
            return float_val
        except ValueError:
            return value
    return value


def create_model(config: Dict):
    ModuleClass = _get_class(config["model"]["class"])
    return ModuleClass(**_convert_value(config["model"]["params"]))


def create_optimizer(config: Dict, model: nn.Module):
    OptimizerClass = _get_class(config["learning"]["optimizer"]["class"])
    return OptimizerClass(
        model.parameters(), **_convert_value(config["learning"]["optimizer"]["params"])
    )


def create_loss(config: Dict):
    if config["learning"]["loss"]["class"].endswith("ComboLoss"):
        from src.losses.ComboLoss import ComboLoss

        losses = []
        for loss_item in config["learning"]["loss"]["params"]["loss_functions"]:
            ClassLoss = _get_class(loss_item["class"])
            losses.append(ClassLoss(**loss_item["params"]))
        return ComboLoss(
            loss_functions=losses,
            weights=_convert_value(config["learning"]["loss"]["params"]["weights"]),
        )
    else:
        LossClass = _get_class(config["learning"]["loss"]["class"])
        return LossClass(**_convert_value(config["learning"]["loss"]["params"]))


def create_metrics(config: Dict):
    metrics = {}
    for metric in config["evaluating"]["metrics"]:
        ClassMetric = _get_class(metric["class"])
        name = metric["name"]
        params = _convert_value(metric["params"])
        metrics[name] = ClassMetric(**params)
    return metrics


def create_scheduler(config: Dict, optimizer: optim.Optimizer):
    SchedulerClass = _get_class(config["learning"]["scheduler"]["class"])
    return SchedulerClass(
        optimizer, **_convert_value(config["learning"]["scheduler"]["params"])
    )
