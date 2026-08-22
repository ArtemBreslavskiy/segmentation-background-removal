import torchmetrics

from src.configs.schemas.evaluating.evaluating import EvaluatingConfig


def create_metrics(config: EvaluatingConfig) -> dict[str, torchmetrics.Metric]:
    metrics = {}
    for metric in config.metrics:
        if metric.name == "iou":
            from torchmetrics import JaccardIndex
            metric_params = metric.model_dump(exclude={"name"})
            metrics["iou"] = JaccardIndex(**metric_params)

        elif metric.name == "accuracy":
            from torchmetrics import Accuracy
            metric_params = metric.model_dump(exclude={"name"})
            metrics["accuracy"] = Accuracy(**metric_params)

        elif metric.name == "precision":
            from torchmetrics import Precision
            metric_params = metric.model_dump(exclude={"name"})
            metrics["precision"] = Precision(**metric_params)

        elif metric.name == "recall":
            from torchmetrics import Recall
            metric_params = metric.model_dump(exclude={"name"})
            metrics["recall"] = Recall(**metric_params)

        elif metric.name == "f1":
            from torchmetrics import F1Score
            metric_params = metric.model_dump(exclude={"name"})
            metrics["f1"] = F1Score(**metric_params)

        elif metric.name == "specificity":
            from torchmetrics import Specificity
            metric_params = metric.model_dump(exclude={"name"})
            metrics["specificity"] = Specificity(**metric_params)

        elif metric.name == "mcc":
            from torchmetrics import MatthewsCorrCoef
            metric_params = metric.model_dump(exclude={"name"})
            metrics["mcc"] = MatthewsCorrCoef(**metric_params)

        else:
            raise ValueError(f"Unknown metric: {metric.name}")

    return metrics
