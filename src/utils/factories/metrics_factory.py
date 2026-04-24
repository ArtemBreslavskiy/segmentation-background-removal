from typing import Dict
from src.utils.factories.factory_utils import get_class, convert_value


def create_metrics(config: Dict):
    metrics = {}
    for metric in config["evaluating"]["metrics"]:
        ClassMetric = get_class(metric["class"])
        name = metric["name"]
        params = convert_value(metric["params"])
        metrics[name] = ClassMetric(**params)
    return metrics