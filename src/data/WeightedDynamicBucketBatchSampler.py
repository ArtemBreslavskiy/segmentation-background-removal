import warnings
from typing import List

import numpy as np
from torch.utils.data import WeightedRandomSampler


class WeightedDynamicBucketBatchSampler(WeightedRandomSampler):
    def __init__(
        self,
        weights: List[float],
        dataset_areas: List[int],
        dataset_aspect_ratios: List[float],
        shuffle: bool = True,
        base_batch_size: int = 8,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        reference_area: int = 512**2,
        replacement: bool = False,
        skip_overload_examples: bool = True,
        send_overload_report: bool = True,
    ):
        if base_batch_size < 1:
            raise ValueError("base_batch_size must be >= 1")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if reference_area < 1:
            raise ValueError("reference_area must be >= 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size cannot be less than min_batch_size")
        if len(weights) == 0:
            raise ValueError("Empty dataset is not allowed")
        if not len(weights) == len(dataset_areas) == len(dataset_aspect_ratios):
            raise ValueError("The length of the weights, dataset_areas, and dataset_aspect_ratio lists does not match")

        super().__init__(weights, len(weights), replacement=replacement)
        self.dataset_areas = np.array(dataset_areas)
        self.dataset_aspect_ratios = np.array(dataset_aspect_ratios)
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.reference_area = reference_area
        self.shuffle = shuffle
        self.send_overload_report = send_overload_report
        self.skip_overload_examples = skip_overload_examples
        self.length = len(weights)

    def __iter__(self):
        sampled_indices = np.array(super().__iter__())
        areas = self.dataset_areas[sampled_indices]
        aspect_ratios = self.dataset_aspect_ratios[sampled_indices]
        # We sort first by aspect ratio, then by area.
        order = np.lexsort((areas, aspect_ratios))
        sampled_indices = sampled_indices[order]
        areas = areas[order]

        batches = []
        max_load = self.base_batch_size * self.reference_area
        i = 0
        missing = 0
        while i < len(sampled_indices):
            current_load = 0
            batch = []
            while i < len(sampled_indices):
                if len(batch) >= self.max_batch_size:
                    break
                if current_load + areas[i] > max_load:
                    if len(batch) == 0:
                        if self.skip_overload_examples:
                            if self.send_overload_report:
                                warnings.warn(
                                    f"An sample with area {areas[i]}"
                                    f" exceeding max_load {max_load} is skipped, even for a batch size of 1"
                                )
                            i += 1
                            missing += 1
                            continue
                        elif self.send_overload_report:
                            warnings.warn(
                                f"An sample was found that exceeds the limits"
                                f" even with a batch size of 1 {areas[i]} > {max_load}"
                            )
                    elif len(batch) < self.min_batch_size:
                        warnings.warn(
                            f"The minimum size batch exceeded the limits, min_batch_size:" f" {self.min_batch_size}"
                        )
                    else:
                        break
                current_load += areas[i]
                batch.append(sampled_indices[i])
                i += 1
            if batch:
                batches.append(batch)
        if self.send_overload_report and self.skip_overload_examples and missing != 0:
            warnings.warn(f"{missing} samples were skipped")

        if self.shuffle:
            np.random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        raise NotImplementedError("Number of batches depends on data distribution and cannot be computed in advance")
