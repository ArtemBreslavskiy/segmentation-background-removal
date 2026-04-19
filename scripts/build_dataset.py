import logging
import shutil
from typing import Optional

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from paths.ProjectPaths import ProjectPaths
from src.logs.logger_setup import configure_loggers


def build_processed_dataset(
    logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("BUILDING PROCESSED DATASET")
    logger.info("=" * 60)

    path = ProjectPaths()
    logger.info("Raw data path: %s", path.RAW_DATA)
    logger.info("Processed data path: %s", path.PROCESSED_DATA)

    if not path.RAW_DATA.exists():
        error_msg = f"Raw data not found at: {path.RAW_DATA}"
        logger.critical("Dataset building failed. Check error log for details.")
        logger.exception(error_msg)
        raise ValueError(error_msg)

    if path.PROCESSED_DATA.exists():
        logger.warning("Processed directory already exists: %s", path.PROCESSED_DATA)
        logger.info("Interactive mode: asking user for confirmation")
        response = input("Delete and recreate? (y/n): ").strip().lower()

        if response.lower() != "y":
            logger.info("Operation cancelled by user")
            return

        logger.info("Cleaning existing processed data...")
        shutil.rmtree(path.PROCESSED_DATA)
        logger.info("Old processed data removed successfully")

    logger.info("Creating directory structure...")

    path.TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    path.TRAIN_MASKS.mkdir(parents=True, exist_ok=True)
    path.VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    path.VAL_MASKS.mkdir(parents=True, exist_ok=True)
    path.TEST_IMAGES.mkdir(parents=True, exist_ok=True)
    path.TEST_MASKS.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading configuration...")
        with open(path.CONFIG) as f:
            config = yaml.safe_load(f)
        logger.debug("Config loaded: %s", config.get("dataset", {}))

        logger.info("Scanning raw data files...")
        train_val_images_lst = list(path.DUTS_TR_IMAGES.glob("*.jpg"))
        train_val_masks_lst = list(path.DUTS_TR_MASKS.glob("*.png"))
        test_images_lst = list(path.DUTS_TE_IMAGES.glob("*.jpg"))
        test_masks_lst = list(path.DUTS_TE_MASKS.glob("*.png"))

        logger.info("Found %d training/validation images", len(train_val_images_lst))
        logger.info("Found %d training/validation masks", len(train_val_masks_lst))
        logger.info("Found %d test images", len(test_images_lst))
        logger.info("Found %d test masks", len(test_masks_lst))

        train_val_images_lst.sort(key=lambda x: x.stem)
        train_val_masks_lst.sort(key=lambda x: x.stem)
        test_images_lst.sort(key=lambda x: x.stem)
        test_masks_lst.sort(key=lambda x: x.stem)

        if len(train_val_images_lst) != len(train_val_masks_lst):
            error_msg = (
                "Mismatch in train/val dataset: "
                f"{len(train_val_images_lst)} images vs "
                f"{len(train_val_masks_lst)} masks"
            )
            logger.critical("Dataset building failed. Check error log for details.")
            logger.exception(error_msg)
            raise ValueError(error_msg)

        if len(test_images_lst) != len(test_masks_lst):
            error_msg = (
                "Mismatch in test dataset: "
                f"{len(test_images_lst)} images vs {len(test_masks_lst)} masks"
            )
            logger.critical("Dataset building failed. Check error log for details.")
            logger.exception(error_msg)
            raise ValueError(error_msg)

        logger.info("Checking filename consistency between images and masks...")
        train_mismatches = []
        test_mismatches = []

        for image, mask in tqdm(
            zip(train_val_images_lst, train_val_masks_lst),
            leave=True,
            desc="Checking train files",
            unit="pair",
        ):
            if image.stem != mask.stem:
                train_mismatches.append((image, mask))
                logger.warning("Mismatch: image %s vs mask %s", image.name, mask.name)

        for image, mask in tqdm(
            zip(test_images_lst, test_masks_lst),
            leave=True,
            desc="Checking test files",
            unit="pair",
        ):
            if image.stem != mask.stem:
                test_mismatches.append((image, mask))
                logger.warning("Mismatch: image %s vs mask %s", image.name, mask.name)

        if len(train_mismatches) > 0 or len(test_mismatches) > 0:
            error_msg = "Filename inconsistencies found in dataset"
            logger.warning(error_msg)

            if len(train_mismatches) > 0:
                logger.error("Train set mismatches: %d pairs", len(train_mismatches))
                for img, mask in train_mismatches[:20]:
                    logger.error("  - %s <-> %s", img.name, mask.name)

            if len(test_mismatches) > 0:
                logger.error("Test set mismatches: %d pairs", len(test_mismatches))
                for img, mask in test_mismatches[:20]:
                    logger.error("  - %s <-> %s", img.name, mask.name)

            raise ValueError(f"{error_msg}. Check error logs for details.")

        logger.info("All filename checks passed ✓")

        logger.info("Splitting dataset into train/val...")
        val_ratio = config["dataset"]["splits"]["ratios"]["val"]
        random_seed = config["dataset"]["splits"]["seed"]

        logger.info("Validation ratio: %.2f, Random seed: %d", val_ratio, random_seed)

        train_images_lst, val_images_lst, train_masks_lst, val_masks_lst = (
            train_test_split(
                train_val_images_lst,
                train_val_masks_lst,
                test_size=val_ratio,
                random_state=random_seed,
                shuffle=True,
            )
        )

        logger.info(
            "Train set: %d images, %d masks",
            len(train_images_lst),
            len(train_masks_lst),
        )
        logger.info(
            "Validation set: %d images, %d masks",
            len(val_images_lst),
            len(val_masks_lst),
        )

        logger.info("Copying test files...")
        for test_image, test_mask in tqdm(
            zip(test_images_lst, test_masks_lst),
            leave=True,
            desc="Copying test data",
            unit="pair",
        ):
            shutil.copy2(test_image, path.TEST_IMAGES / test_image.name)
            shutil.copy2(test_mask, path.TEST_MASKS / test_mask.name)

        logger.info("Copying train files...")
        for train_image, train_mask in tqdm(
            zip(train_images_lst, train_masks_lst),
            leave=True,
            desc="Copying train data",
            unit="pair",
        ):
            shutil.copy2(train_image, path.TRAIN_IMAGES / train_image.name)
            shutil.copy2(train_mask, path.TRAIN_MASKS / train_mask.name)

        logger.info("Copying validation files...")
        for val_image, val_mask in tqdm(
            zip(val_images_lst, val_masks_lst),
            leave=True,
            desc="Copying validation data",
            unit="pair",
        ):
            shutil.copy2(val_image, path.VAL_IMAGES / val_image.name)
            shutil.copy2(val_mask, path.VAL_MASKS / val_mask.name)

        logger.info("=" * 60)
        logger.info("DATASET BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info("Train images: %d", len(train_images_lst))
        logger.info("Validation images: %d", len(val_images_lst))
        logger.info("Test images: %d", len(test_images_lst))
        logger.info(
            "Total: %d",
            len(train_images_lst) + len(test_images_lst) + len(val_images_lst),
        )
        logger.info("=" * 60)

    except Exception as ex:
        logger.exception("Error building dataset: %s", ex)
        raise


if __name__ == "__main__":
    from src.logs.logger_setup import get_logger

    path = ProjectPaths()
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    configure_loggers(path.CONFIG, path.LOGS)
    build_processed_dataset(logger=get_logger(config["logs"]["types"]["data"]["name"]))
