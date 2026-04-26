import logging
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import yaml
import json
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from ProjectPaths import ProjectPaths
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
    with open(path.CONFIG) as f:
        config = yaml.safe_load(f)

    if not path.RAW_DATA.exists():
        error_msg = f"Raw data not found at: {path.RAW_DATA}"
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
    path.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    datasets_paths = [f for f in path.RAW_DATA.iterdir() if f.is_dir()]
    datasets_names = [f.name for f in datasets_paths]
    if not datasets_names:
        error_msg = "Raw data directory does not contain any dataset folders."
        logger.exception(error_msg)
        raise ValueError(error_msg)
    elif len(datasets_names) == 1:
        msg = f"Dataset name: {datasets_names[0]}"
    else:
        msg = "The multi-dataset contains: " + ", ".join(datasets_names) + "."
    logger.info(msg)

    def search_correct_directories(path: Path, criteria: Callable[[Path], bool]) -> List[Path]:
        if not path.is_dir():
            return []
        correct_directories = []
        if criteria(path):
            correct_directories.append(path)
        else:
            paths = [f for f in path.iterdir() if f.is_dir()]
            for p in paths:
                correct_directories.extend(search_correct_directories(p, criteria))
        return correct_directories

    def image_directory_criteria(path: Path) -> bool:
        name = path.name.lower()
        keywords = {'image', 'img', 'im', 'images', 'picture', 'photo', 'original'}
        return any(kw in name for kw in keywords)

    def mask_directory_criteria(path: Path) -> bool:
        name = path.name.lower()
        keywords = {
            'mask', 'masks', 'matt', 'matte', 'label', 'labels',
            'class', 'classes', 'object', 'gt', 'groundtruth',
            'ground_truth', 'seg', 'segmentation', 'annotation'
        }
        return any(kw in name for kw in keywords)

    def build_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
        pairs = []
        mask_dict = {}
        missing = 0
        for mask_path in mask_dir.rglob("*"):
            if mask_path.is_file() and mask_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                mask_dict[mask_path.stem] = mask_path
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                stem = image_path.stem
                if stem in mask_dict:
                    mask_path = mask_dict[stem]
                    pairs.append((image_path, mask_path))
                else:
                    missing += 1
        used_masks = set(p[1] for p in pairs)
        unused_masks = len(mask_dict) - len(used_masks)
        logger.info(f"Unused masks (no corresponding image): {unused_masks}")
        logger.info(f"Total pairs: {len(pairs)}, missing: {missing}")
        return pairs

    def validate_pair(image_dir: Path, mask_dir: Path) -> bool:
        image_stems = {p.stem for p in image_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        mask_stems = {p.stem for p in mask_dir.rglob("*")
                      if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        return not image_stems.isdisjoint(mask_stems)

    def get_image_shape(image_path: Path):
        try:
            with Image.open(image_path) as img:
                return img.height, img.width
        except Exception as ex:
            return None

    try:
        logger.info("Data structure recognition...")
        datasets = {}
        for dataset_path in datasets_paths:
            image_dirs = search_correct_directories(dataset_path, image_directory_criteria)
            mask_dirs = search_correct_directories(dataset_path, mask_directory_criteria)
            if len(image_dirs) != len(mask_dirs):
                logger.warning(
                    f"Mismatch in {dataset_path.name}: {len(image_dirs)} image dirs vs {len(mask_dirs)} mask dirs. "
                    f"Will pair only the first {min(len(image_dirs), len(mask_dirs))}."
                )
            datasets[dataset_path.name] = []
            used_masks = set()
            for image_path in image_dirs:
                for mask_path in mask_dirs:
                    if mask_path not in used_masks:
                        if validate_pair(image_path, mask_path):
                            file_pairs = build_pairs(image_path, mask_path)
                            if file_pairs:
                                datasets[dataset_path.name].extend(file_pairs)
                                used_masks.add(mask_path)
                                logger.info("%s + %s -> %d file pairs", image_path.name, mask_path.name, len(file_pairs))
                                break
                        else:
                            logger.debug(f"Skipped pair: {image_path.name} + {mask_path.name} (no common files)")
            if not datasets[dataset_path.name]:
                logger.warning("No file pairs were built for dataset %s. Check directory structure and file names.", dataset_path.name)

        for name, pairs in datasets.items():
            logger.debug("%s contains %d pairs", name, len(pairs))
        data_paths_lst = [(image_path, mask_path, source) for source, pair in datasets.items() for image_path, mask_path in pair]

        logger.info("Splitting dataset into train/val...")
        random_seed = config["dataset"]["splits"]["seed"]
        test_ratio = config["dataset"]["splits"]["ratios"]["test"]
        val_ratio = config["dataset"]["splits"]["ratios"]["val"]
        test_val_ratio = test_ratio + val_ratio

        sources = [s for _, _, s in data_paths_lst]
        train, test_val = train_test_split(
            data_paths_lst,
            test_size=test_val_ratio,
            random_state=random_seed,
            stratify=sources,
        )
        test_val_sources = [s for _, _, s in test_val]
        val, test = train_test_split(
            test_val,
            test_size=test_ratio/test_val_ratio,
            random_state=random_seed,
            stratify=test_val_sources,
        )
        logger.info("Train: %d, Val: %d, Test: %d", len(train), len(val), len(test))

        def save_manifest(data: List[Tuple[Path, Path, str]], filepath: Path):
            manifest = []
            skipped = 0
            for d in tqdm(data, desc=f"Saving {filepath.name}", leave=True):
                image_path, mask_path, source = d
                resolution = get_image_shape(image_path)
                if resolution is None:
                    skipped += 1
                    continue
                manifest.append({
                    "image": str(image_path.resolve()),
                    "mask": str(mask_path.resolve()),
                    "source": source,
                    "resolution": resolution,
                })
            if skipped > 0:
                logger.warning(f"Skipped {skipped} pairs due to image read errors in {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info("Creating JSON files...")
        save_manifest(train, path.TRAIN)
        save_manifest(test, path.TEST)
        save_manifest(val, path.VAL)
        logger.debug("JSON files have been created successfully")

        logger.info("=" * 60)
        logger.info("DATASET BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info("Train images: %d", len(train))
        logger.info("Validation images: %d", len(val))
        logger.info("Test images: %d", len(test))
        logger.info(
            "Total: %d",
            len(train) + len(val) + len(test)
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
