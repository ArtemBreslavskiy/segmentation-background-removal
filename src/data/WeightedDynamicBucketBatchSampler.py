import numpy as np
import warnings
from torch.utils.data import WeightedRandomSampler


class WeightedDynamicBucketBatchSampler(WeightedRandomSampler):
    def __init__(
        self,
        weights: list[float],
        dataset_areas: list[int],
        dataset_aspect_ratios: list[float],
        shuffle: bool = True,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        max_load: int = 2097152,
        replacement: bool = False,
        skip_overload_examples: bool = True,
        send_overload_report: bool = True,
    ):
        self._validate_weights(weights)
        self._validate_batch_sizes(max_batch_size, min_batch_size)
        self._validate_max_load(max_load)
        self._validate_lengths(weights, dataset_areas, dataset_aspect_ratios)

        super().__init__(weights, len(weights), replacement=replacement)
        self.dataset_areas = np.array(dataset_areas)
        self.dataset_aspect_ratios = np.array(dataset_aspect_ratios)
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_load = max_load
        self.shuffle = shuffle
        self.send_overload_report = send_overload_report
        self.skip_overload_examples = skip_overload_examples

    @staticmethod
    def _validate_weights(weights) -> None:
        if len(weights) == 0:
            raise ValueError("Empty dataset is not allowed")

    @staticmethod
    def _validate_batch_sizes(max_batch_size: int, min_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size cannot be less than min_batch_size")

    @staticmethod
    def _validate_max_load(max_load: int) -> None:
        if max_load < 1:
            raise ValueError("max_load must be >= 1")

    @staticmethod
    def _validate_lengths(weights: list[float], dataset_areas: list[int], dataset_aspect_ratios: list[float]) -> None:
        if not len(weights) == len(dataset_areas) == len(dataset_aspect_ratios):
            raise ValueError("The length of the weights, dataset_areas, and dataset_aspect_ratio lists does not match")

    def __iter__(self):
        sampled_indices = np.array(list(super().__iter__()))
        areas = self.dataset_areas[sampled_indices]
        aspect_ratios = self.dataset_aspect_ratios[sampled_indices]
        # We sort first by aspect ratio, then by area.
        order = np.lexsort((areas, aspect_ratios))
        sampled_indices = sampled_indices[order]
        areas = areas[order]

        batches = []
        current_index = 0
        missing = 0

        while current_index < len(sampled_indices):
            current_load = 0
            batch = []

            while current_index < len(sampled_indices):
                if len(batch) >= self.max_batch_size:
                    break

                if current_load + areas[current_index] > self.max_load:
                    if len(batch) == 0:
                        if self.send_overload_report:
                            warnings.warn(
                                f"An sample was found that exceeds the limits"
                                f" even with a batch size of 1 {areas[current_index]} > {self.max_load}"
                            )
                        if self.skip_overload_examples:
                            current_index += 1
                            missing += 1
                            continue

                    elif len(batch) < self.min_batch_size:
                        warnings.warn(
                            f"The minimum size batch exceeded the limits, min_batch_size:" f" {self.min_batch_size}"
                        )
                    else:
                        break

                current_load += areas[current_index]
                batch.append(sampled_indices[current_index])
                current_index += 1

            if batch:
                batches.append(batch)

        if self.send_overload_report and self.skip_overload_examples and missing != 0:
            warnings.warn(f"{missing} samples were skipped")

        if self.shuffle:
            np.random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return 1
