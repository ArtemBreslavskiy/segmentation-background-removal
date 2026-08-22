from src.configs.schemas.dataloader.sampler import BaseSamplerConfig
from src.data.BinarySegmentationDataset import BinarySegmentationDataset
from src.utils.sampler_utils import get_area_and_aspect_ratio, get_sample_weight


def create_batch_sampler(config: BaseSamplerConfig, dataset: BinarySegmentationDataset):
    if config.type == "wdb":
        from src.data.WeightedDynamicBucketBatchSampler import WeightedDynamicBucketBatchSampler

        manifest = dataset.get_manifest_with_correct_resolution()
        weights = []
        dataset_areas = []
        dataset_aspect_ratios = []
        for item in manifest:
            weights.append(get_sample_weight(config.weights, item["source"]))
            h, w = item["resolution"]
            area, aspect_ratio = get_area_and_aspect_ratio(h, w)
            dataset_areas.append(area)
            dataset_aspect_ratios.append(aspect_ratio)

        sampler_params = config.model_dump(exclude={"type"})
        sampler_params["weights"] = weights
        sampler_params["dataset_areas"] = dataset_areas
        sampler_params["dataset_aspect_ratios"] = dataset_aspect_ratios
        return WeightedDynamicBucketBatchSampler(**sampler_params)

    else:
        raise ValueError(f"Unknown batch sampler type: {config.type}")
