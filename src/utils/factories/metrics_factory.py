from typing import Dict

from src.utils.factories.factory_utils import convert_value, get_class


def create_metrics(config: Dict):
    metrics = {}
    for metric in config["evaluating"]["metrics"]:
        ClassMetric = get_class(metric["class"])
        name = metric["name"]
        params = convert_value(metric["params"])
        metrics[name] = ClassMetric(**params)
    return metrics
