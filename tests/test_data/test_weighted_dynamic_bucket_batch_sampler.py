import pytest
import torch
import numpy as np
import cv2
import json
from pathlib import Path

from src.data.WeightedDynamicBucketBatchSampler import WeightedDynamicBucketBatchSampler


class TestWeightedDynamicBucketBatchSampler:
    @pytest.mark.parametrize("base_batch_size", [2, 4, 6])
    @pytest.mark.parametrize("max_batch_size", [4, 5, 6])
    @pytest.mark.parametrize("min_batch_size", [1, 2, 3])
    @pytest.mark.parametrize("reference_area", [2048, 4096])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("send_overload_report", [True, False])
    @pytest.mark.parametrize("skip_overload_examples", [True, False])
    def test_init(
        self,
        base_batch_size,
        max_batch_size,
        min_batch_size,
        reference_area,
        shuffle,
        send_overload_report,
        skip_overload_examples,
    ):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        sampler = WeightedDynamicBucketBatchSampler(
            weights=weights,
            dataset_areas=dataset_areas,
            dataset_aspect_ratios=dataset_aspect_ratios,
            base_batch_size=base_batch_size,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            reference_area=reference_area,
            shuffle=shuffle,
            send_overload_report=send_overload_report,
            skip_overload_examples=skip_overload_examples,
        )
        assert isinstance(sampler.dataset_areas, np.ndarray)
        assert isinstance(sampler.dataset_aspect_ratios, np.ndarray)
        assert sampler.length == 3
        assert sampler.base_batch_size == base_batch_size
        assert sampler.max_batch_size == max_batch_size
        assert sampler.min_batch_size == min_batch_size
        assert sampler.reference_area == reference_area
        assert sampler.shuffle == shuffle
        assert sampler.send_overload_report == send_overload_report
        assert sampler.skip_overload_examples == skip_overload_examples

    @pytest.mark.parametrize("base_batch_size", [0, -1])
    def test_init_with_invalid_base_batch_size(self, base_batch_size):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="base_batch_size must be >= 1"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False,
                base_batch_size=base_batch_size
            )

    @pytest.mark.parametrize("max_batch_size", [0, -1])
    def test_init_with_invalid_max_batch_size(self, max_batch_size):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False,
                max_batch_size=max_batch_size
            )

    @pytest.mark.parametrize("min_batch_size", [0, -1])
    def test_init_with_invalid_min_batch_size(self, min_batch_size):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="min_batch_size must be >= 1"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False,
                min_batch_size=min_batch_size
            )

    @pytest.mark.parametrize("reference_area", [0, -1])
    def test_init_with_invalid_reference_area(self, reference_area):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="reference_area must be >= 1"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False,
                reference_area=reference_area
            )

    def test_init_with_non_matching_length(self):
        weights = [1.0, 1.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="The length of the weights, dataset_areas, "
                                             "and dataset_aspect_ratio lists does not match"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False
            )

    def test_init_with_not_possible_min_max_batch_size(self):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        with pytest.raises(ValueError, match="max_batch_size cannot be less than min_batch_size"):
            WeightedDynamicBucketBatchSampler(
                weights=weights,
                dataset_areas=dataset_areas,
                dataset_aspect_ratios=dataset_aspect_ratios,
                send_overload_report=False,
                max_batch_size=8,
                min_batch_size=16
            )

    def test_iter(self):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        sampler = WeightedDynamicBucketBatchSampler(
            weights=weights,
            dataset_areas=dataset_areas,
            dataset_aspect_ratios=dataset_aspect_ratios,
            send_overload_report=False,
            shuffle=False,
            replacement=False,
            max_batch_size=1
        )
        batches = []
        for batch in sampler:
            batches.append(batch)
        idx0 = batches[0][0]
        idx1 = batches[1][0]
        idx2 = batches[2][0]
        assert dataset_aspect_ratios[idx0] < dataset_aspect_ratios[idx1] < dataset_aspect_ratios[idx2]

    @pytest.mark.parametrize("replacement", [True, False])
    @pytest.mark.parametrize("shuffle", [True, False])
    def test_iter_max_image_area(self, replacement, shuffle):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        sampler = WeightedDynamicBucketBatchSampler(
            weights=weights,
            dataset_areas=dataset_areas,
            dataset_aspect_ratios=dataset_aspect_ratios,
            send_overload_report=False,
            shuffle=shuffle,
            replacement=replacement,
            max_batch_size=1,
            base_batch_size=1,
            reference_area=1500
        )
        batches = []
        for batch in sampler:
            batches.append(batch)
        all_indices = [i for b in batches for i in b]
        assert 2 not in all_indices

    @pytest.mark.parametrize("replacement", [True, False])
    @pytest.mark.parametrize("shuffle", [True, False])
    def test_iter_batch_area(self, replacement, shuffle):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [1024, 1024, 1024]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        sampler = WeightedDynamicBucketBatchSampler(
            weights=weights,
            dataset_areas=dataset_areas,
            dataset_aspect_ratios=dataset_aspect_ratios,
            send_overload_report=False,
            shuffle=shuffle,
            replacement=replacement,
            max_batch_size=8,
            base_batch_size=1,
            reference_area=2500
        )
        batches = []
        for batch in sampler:
            batches.append(batch)
        batch1 = batches[0]
        batch2 = batches[1]
        area1 = 0
        for inx in batch1:
            area1 += dataset_areas[inx]
        area2 = 0
        for inx in batch2:
            area2 += dataset_areas[inx]

        assert len(batches) == 2
        assert area1 < 2500
        assert area2 < 2500

    def test_warning_when_heavy_skipped(self):
        sampler = WeightedDynamicBucketBatchSampler(
            weights=[1.0],
            dataset_areas=[5000],
            dataset_aspect_ratios=[1.0],
            base_batch_size=1,
            reference_area=100,
            skip_overload_examples=True,
            send_overload_report=True,
            max_batch_size=1,
            min_batch_size=1,
            shuffle=False
        )
        with pytest.warns(UserWarning) as record:
            batches = []
            for batch in sampler:
                batches.append(batch)

        assert len(record) == 2
        assert "An sample with area 5000 exceeding max_load 100 is skipped" in str(record[0].message)
        assert "1 samples were skipped" in str(record[1].message)

    def test_len(self):
        weights = [1.0, 1.5, 0.5]
        dataset_areas = [512, 1024, 2048]
        dataset_aspect_ratios = [0.7, 0.3, 1.5]
        sampler = WeightedDynamicBucketBatchSampler(
            weights=weights,
            dataset_areas=dataset_areas,
            dataset_aspect_ratios=dataset_aspect_ratios,
            send_overload_report=False,
        )
        with pytest.raises(NotImplementedError, match="Number of batches depends on data distribution "
                                                      "and cannot be computed in advance"):
            len(sampler)
