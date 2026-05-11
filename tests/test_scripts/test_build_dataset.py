import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from scripts.build_dataset import (
    _build_pairs,
    _get_image_shape,
    _image_directory_criteria,
    _mask_directory_criteria,
    _save_manifest,
    _search_correct_directories,
    _validate_pair,
    build_processed_dataset,
)


class TestBuildDataset:
    def test_search_correct_images_directories(self, tmp_path):
        img_dir = tmp_path / "root"
        img_1 = img_dir / "images"
        img_2 = img_dir / "data" / "images"
        img_3 = img_dir / "data1" / "data2" / "data3" / "images"
        img_1.mkdir(parents=True)
        img_2.mkdir(parents=True)
        img_3.mkdir(parents=True)

        result = _search_correct_directories(img_dir, _image_directory_criteria)
        assert set(result) == {img_1, img_2, img_3}

    def test_search_correct_masks_directories(self, tmp_path):
        mask_dir = tmp_path / "root"
        mask_1 = mask_dir / "masks"
        mask_2 = mask_dir / "data" / "masks"
        mask_3 = mask_dir / "data1" / "data2" / "data3" / "masks"
        mask_1.mkdir(parents=True)
        mask_2.mkdir(parents=True)
        mask_3.mkdir(parents=True)

        result = _search_correct_directories(mask_dir, _mask_directory_criteria)
        assert set(result) == {mask_1, mask_2, mask_3}

    def test_build_pairs(self, tmp_path, dummy_logger, caplog):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img1 = img_dir / "121212.jpg"
        img1.touch()
        img2 = img_dir / "13131.jpg"
        img2.touch()
        img3 = img_dir / "1414.jpg"
        img3.touch()
        img4 = img_dir / "151.jpg"
        img4.touch()

        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        mask1 = mask_dir / "121212.jpg"
        mask1.touch()
        mask2 = mask_dir / "13131.jpg"
        mask2.touch()
        mask3 = mask_dir / "1414.jpg"
        mask3.touch()
        mask4 = mask_dir / "151.jpg"
        mask4.touch()

        pairs = _build_pairs(img_dir, mask_dir, dummy_logger)
        assert set(pairs) == {(img1, mask1), (img2, mask2), (img3, mask3), (img4, mask4)}
        assert any(
            rec.levelname == "INFO" and "Unused masks (no corresponding image)" in rec.message for rec in caplog.records
        )
        assert any(rec.levelname == "INFO" and "Missing images: 0" in rec.message for rec in caplog.records)
        assert any(rec.levelname == "INFO" and "Total pairs: 4" in rec.message for rec in caplog.records)

    def test_validate_pair_positive(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "121212.jpg").touch()

        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        (mask_dir / "121212.png").touch()

        assert _validate_pair(img_dir, mask_dir) == True

    def test_validate_pair_negative(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "121212.jpg").touch()

        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        (mask_dir / "111111.png").touch()

        assert _validate_pair(img_dir, mask_dir) == False

    def test_get_image_shape(self, tmp_path):
        img_path = tmp_path / "image.jpg"
        Image.new("RGB", (10, 20)).save(img_path)
        h, w = _get_image_shape(img_path)
        assert h == 20
        assert w == 10

    def test_get_image_shape_invalid(self, tmp_path):
        img_path = tmp_path / "broken.jpg"
        h, w = _get_image_shape(img_path)
        assert h == None
        assert w == None

    def test_save_manifest(self, tmp_path, dummy_logger, caplog):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        data = []
        images = set()
        masks = set()
        sources = {"Dataset1", "Dataset2"}
        for i in range(4):
            img = img_dir / f"00000{i}.jpg"
            Image.new("RGB", (10, 20)).save(img)
            mask = mask_dir / f"00000{i}.jpg"
            Image.new("1", (10, 20)).save(mask)

            data.append((img, mask, "Dataset1"))
            images.add(img)
            masks.add(mask)

        data.append((img_dir / "000004.jpg", mask_dir / "000004.jpg", "Dataset1"))

        for i in range(5, 11):
            img = img_dir / f"00000{i}.jpg"
            Image.new("RGB", (10, 20)).save(img)
            mask = mask_dir / f"00000{i}.jpg"
            Image.new("1", (10, 20)).save(mask)

            data.append((img, mask, "Dataset2"))
            images.add(img)
            masks.add(mask)

        json_path = tmp_path / "manifest.json"
        _save_manifest(data, json_path, dummy_logger)

        assert json_path.exists()
        with open(json_path, "r") as f:
            manifest = json.load(f)

        assert len(manifest) == 10
        for item in manifest:
            assert Path(item["image"]) in images
            assert Path(item["mask"]) in masks
            assert item["source"] in sources
            assert item["resolution"] == [20, 10]

            assert isinstance(item["image"], str)
            assert isinstance(item["mask"], str)
            assert isinstance(item["source"], str)
            assert isinstance(item["resolution"], list)

        assert any(
            rec.levelname == "WARNING" and "Skipped 1 pairs due to image read errors in" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.parametrize("processed_exist", [True, False])
    def test_build_successful(self, dummy_config, dummy_logger, processed_exist, tmp_path, caplog, monkeypatch):
        raw_path = tmp_path / "raw"
        processed_path = tmp_path / "processed"
        if processed_exist:
            processed_path.mkdir()
        train_path = processed_path / "train.json"
        test_path = processed_path / "test.json"
        val_path = processed_path / "val.json"

        paths = MagicMock()
        paths.RAW_DATA = raw_path
        paths.PROCESSED_DATA = processed_path
        paths.TRAIN = train_path
        paths.TEST = test_path
        paths.VAL = val_path

        dataset1_dir = raw_path / "Dataset1"
        img_dir1 = dataset1_dir / "image"
        mask_dir1 = dataset1_dir / "mask"
        img_dir1.mkdir(parents=True)
        mask_dir1.mkdir(parents=True)

        for i in range(12):
            img = img_dir1 / f"00000{i}.jpg"
            Image.new("RGB", (10, 20 + i)).save(img)
            mask = mask_dir1 / f"00000{i}.jpg"
            Image.new("1", (10, 20 + i)).save(mask)

        img = img_dir1 / "0000012.jpg"
        Image.new("RGB", (10, 20)).save(img)

        dataset2_dir = raw_path / "Dataset2"
        img_dir2 = dataset2_dir / "image"
        mask_dir2 = dataset2_dir / "mask"
        img_dir2.mkdir(parents=True)
        mask_dir2.mkdir(parents=True)

        for i in range(13, 21):
            img = img_dir2 / f"00000{i}.jpg"
            Image.new("RGB", (10, 20 + i)).save(img)
            mask = mask_dir2 / f"00000{i}.jpg"
            Image.new("1", (10, 20 + i)).save(mask)

        with patch("builtins.input", return_value="y"):
            with patch("scripts.build_dataset.ProjectPaths", return_value=paths):
                build_processed_dataset(dummy_config, dummy_logger)

        assert processed_path.exists()
        assert train_path.exists()
        assert test_path.exists()
        assert val_path.exists()

        with open(paths.TRAIN, "r") as f:
            train_manifest = json.load(f)
        with open(paths.TEST, "r") as f:
            test_manifest = json.load(f)
        with open(paths.VAL, "r") as f:
            val_manifest = json.load(f)

        assert len(train_manifest) == 16
        assert len(test_manifest) == 2
        assert len(val_manifest) == 2

        for manifest in [train_manifest, test_manifest, val_manifest]:
            for item in manifest:
                assert "image" in item
                assert "mask" in item
                assert item["source"] in {"Dataset1", "Dataset2"}
                assert "resolution" in item
                assert Path(item["image"]).exists()
                assert Path(item["mask"]).exists()

        assert any(rec.levelname == "INFO" and "BUILDING PROCESSED DATASET" in rec.message for rec in caplog.records)
        assert any(rec.levelname == "INFO" and f"Raw data path: {raw_path}" in rec.message for rec in caplog.records)
        assert any(
            rec.levelname == "INFO" and f"Processed data path: {processed_path}" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "INFO" and "The multi-dataset contains: Dataset1, Dataset2." in rec.message
            for rec in caplog.records
        )
        assert any(rec.levelname == "INFO" and "Data structure recognition" in rec.message for rec in caplog.records)
        assert any(rec.levelname == "INFO" and "Missing images: 1" in rec.message for rec in caplog.records)
        assert any(
            rec.levelname == "INFO" and "Splitting dataset into train/test/val..." in rec.message
            for rec in caplog.records
        )
        assert any(rec.levelname == "INFO" and "Creating JSON files" in rec.message for rec in caplog.records)
        assert any(
            rec.levelname == "DEBUG" and "JSON files have been created successfully" in rec.message
            for rec in caplog.records
        )
        assert any(
            rec.levelname == "INFO" and "DATASET BUILD COMPLETED SUCCESSFULLY" in rec.message for rec in caplog.records
        )
        if processed_exist:
            assert any(
                rec.levelname == "WARNING" and f"Processed directory already exists: {processed_path}" in rec.message
                for rec in caplog.records
            )
            assert any(
                rec.levelname == "INFO" and "Interactive mode: asking user for confirmation" in rec.message
                for rec in caplog.records
            )
            assert any(
                rec.levelname == "INFO" and "Cleaning existing processed data" in rec.message for rec in caplog.records
            )
            assert any(
                rec.levelname == "INFO" and "Old processed data removed successfully" in rec.message
                for rec in caplog.records
            )

    def test_missing_raw_data(self, tmp_path, dummy_config, dummy_logger, caplog):
        paths = MagicMock()
        paths.RAW_DATA = tmp_path / "nonexistent"
        with patch("scripts.build_dataset.ProjectPaths", return_value=paths):
            with pytest.raises(ValueError, match="Raw data not found"):
                build_processed_dataset(config=dummy_config, logger=dummy_logger)
        assert any(rec.levelname == "ERROR" and "Raw data not found" in rec.message for rec in caplog.records)

    def test_empty_raw_data(self, tmp_path, dummy_config, dummy_logger, caplog):
        raw = tmp_path / "raw"
        raw.mkdir()
        paths = MagicMock()
        paths.RAW_DATA = raw
        paths.PROCESSED_DATA = tmp_path / "processed"
        with patch("scripts.build_dataset.ProjectPaths", return_value=paths):
            with pytest.raises(ValueError, match="Raw data directory does not contain any dataset folders."):
                build_processed_dataset(config=dummy_config, logger=dummy_logger)
        assert any(
            rec.levelname == "ERROR" and "Raw data directory does not contain any dataset folders." in rec.message
            for rec in caplog.records
        )

    def test_user_skip_rebuild(self, tmp_path, dummy_config, dummy_logger, caplog):
        raw = tmp_path / "raw"
        raw.mkdir()
        processed = tmp_path / "processed"
        processed.mkdir()
        paths = MagicMock()
        paths.RAW_DATA = raw
        paths.PROCESSED_DATA = processed
        with patch("builtins.input", return_value="n"):
            with patch("scripts.build_dataset.ProjectPaths", return_value=paths):
                build_processed_dataset(config=dummy_config, logger=dummy_logger)
        assert processed.exists()
        assert any(rec.levelname == "INFO" and "Operation cancelled by user" in rec.message for rec in caplog.records)

    def test_build_with_one_empty_dataset(self, dummy_config, dummy_logger, tmp_path, caplog):
        raw_path = tmp_path / "raw"
        processed_path = tmp_path / "processed"
        train_path = processed_path / "train.json"
        test_path = processed_path / "test.json"
        val_path = processed_path / "val.json"

        paths = MagicMock()
        paths.RAW_DATA = raw_path
        paths.PROCESSED_DATA = processed_path
        paths.TRAIN = train_path
        paths.TEST = test_path
        paths.VAL = val_path

        dataset1_dir = raw_path / "Dataset1"
        img_dir1 = dataset1_dir / "image"
        mask_dir1 = dataset1_dir / "mask"
        img_dir1.mkdir(parents=True)
        mask_dir1.mkdir(parents=True)

        for i in range(10):
            img = img_dir1 / f"00000{i}.jpg"
            Image.new("RGB", (10, 20)).save(img)
            mask = mask_dir1 / f"00000{i}.jpg"
            Image.new("1", (10, 20)).save(mask)

        dataset2_dir = raw_path / "Dataset2"
        dataset2_dir.mkdir()

        with patch("builtins.input", return_value="y"):
            with patch("scripts.build_dataset.ProjectPaths", return_value=paths):
                build_processed_dataset(dummy_config, dummy_logger)

        assert any(
            rec.levelname == "WARNING"
            and "No file pairs were built for dataset Dataset2. "
            "Check directory structure and file names." in rec.message
            for rec in caplog.records
        )
