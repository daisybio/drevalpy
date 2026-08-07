"""Load cross-validation split CSV files into a DrugResponseDataset."""

from __future__ import annotations

from pathlib import Path

from .dataset import DrugResponseDataset


def _list_cv_split_files(path: Path) -> list[str]:
    # Filenames stay ``str``: the predicates below and the sort that drives the
    # train/test pairing in ``_load_train_test_splits`` operate on names, not paths.
    files = [
        entry.name for entry in path.iterdir() if entry.name.endswith(".csv") and entry.name.startswith("cv_split")
    ]
    if not files:
        raise AssertionError(f"No cv split files found in {path}")
    return files


def _partition_split_filenames(files: list[str]) -> dict[str, list[str]]:
    validation_es_splits = [file for file in files if "validation_es" in file]
    validation_splits = [file for file in files if "validation" in file and file not in validation_es_splits]
    partitions = {
        "train": [file for file in files if "train" in file],
        "test": [file for file in files if "test" in file],
        "validation": validation_splits,
        "validation_es": validation_es_splits,
        "early_stopping": [file for file in files if "early_stopping" in file],
    }
    for names in partitions.values():
        names.sort()
    return partitions


def _load_train_test_splits(
    path: Path,
    dataset_name: str,
    train_splits: list[str],
    test_splits: list[str],
) -> list[dict[str, DrugResponseDataset]]:
    cv_splits: list[dict[str, DrugResponseDataset]] = []
    for split_train, split_test in zip(train_splits, test_splits, strict=True):
        tr_split = DrugResponseDataset.from_csv(path / split_train, dataset_name=dataset_name)
        te_split = DrugResponseDataset.from_csv(path / split_test, dataset_name=dataset_name)
        cv_splits.append({"train": tr_split, "test": te_split})
    return cv_splits


def _attach_optional_split_modes(
    path: Path,
    dataset_name: str,
    cv_splits: list[dict[str, DrugResponseDataset]],
    optional_splits: dict[str, list[str]],
) -> None:
    for mode, filenames in optional_splits.items():
        if not filenames:
            continue
        for i, filename in enumerate(filenames):
            split = DrugResponseDataset.from_csv(path / filename, dataset_name=dataset_name)
            cv_splits[i][mode] = split


def load_cv_splits_from_dir(path: str | Path, dataset_name: str) -> list[dict[str, DrugResponseDataset]]:
    """Load train/test and optional validation splits from a split directory.

    :param path: Directory containing ``cv_split*.csv`` files.
    :param dataset_name: Dataset label passed to ``DrugResponseDataset.from_csv``.
    :returns: List of split dicts keyed by role names such as ``train`` and ``test``.
    """
    splits_dir = Path(path)
    files = _list_cv_split_files(splits_dir)
    partitions = _partition_split_filenames(files)
    cv_splits = _load_train_test_splits(splits_dir, dataset_name, partitions["train"], partitions["test"])
    optional = {key: partitions[key] for key in ("validation", "validation_es", "early_stopping")}
    _attach_optional_split_modes(splits_dir, dataset_name, cv_splits, optional)
    return cv_splits
