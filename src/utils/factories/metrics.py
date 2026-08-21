from src.configs.schemas.evaluating.evaluating import EvaluatingConfig


def create_metrics(config: EvaluatingConfig) -> dict[str, ]:
    metrics = {}
    for metric in config.metrics:
        if config.name == "iou":
            from torchmetrics import JaccardIndex
            metric_params = metric.model_dump(exclude={"name"})
            metrics["iou"] = JaccardIndex(**metric_params)

        elif config.name == "accuracy":
            from torchmetrics import Accuracy
            metric_params = metric.model_dump(exclude={"name"})
            metrics["accuracy"] = Accuracy(**metric_params)

        elif config.name == "precision":
            from torchmetrics import Precision
            metric_params = metric.model_dump(exclude={"name"})
            metrics["precision"] = Precision(**metric_params)

        elif config.name == "recall":
            from torchmetrics import Recall
            metric_params = metric.model_dump(exclude={"name"})
            metrics["recall"] = Recall(**metric_params)

        elif config.name == "f1":
            from torchmetrics import F1Score
            metric_params = metric.model_dump(exclude={"name"})
            metrics["f1"] = F1Score(**metric_params)

        elif config.name == "specificity":
            from torchmetrics import Specificity
            metric_params = metric.model_dump(exclude={"name"})
            metrics["specificity"] = Specificity(**metric_params)

        elif config.name == "mcc":
            from torchmetrics import MatthewsCorrCoef
            metric_params = metric.model_dump(exclude={"name"})
            metrics["mcc"] = MatthewsCorrCoef(**metric_params)

        else:
            raise ValueError(f"Unknown metric type: {config.type}")

    return metrics
