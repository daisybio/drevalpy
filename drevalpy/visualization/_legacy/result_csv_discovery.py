"""Discover prediction CSV files under the experiment results layout."""

from __future__ import annotations

import pathlib

_RESULT_CATEGORIES = ("predictions", "cross_study", "randomization", "robustness")


def _csv_files_in_category(algorithm_dir: pathlib.Path, category: str) -> list[pathlib.Path]:
    category_dir = algorithm_dir / category
    if not category_dir.is_dir():
        return []
    return sorted(category_dir.glob("*.csv"))


def discover_result_csv_files(result_dir: pathlib.Path, dataset: str) -> list[pathlib.Path]:
    """Collect prediction CSV files from the experiment directory layout.

    Expected layout: ``{result_dir}/{dataset}/{split_label}/{algorithm}/{category}/*.csv``.

    :param result_dir: Root experiment results directory.
    :param dataset: Dataset subdirectory name.

    :returns: Sorted list of discovered prediction CSV paths.
    """
    dataset_dir = result_dir / dataset
    if not dataset_dir.is_dir():
        return []

    result_files: list[pathlib.Path] = []
    for split_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for algorithm_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if algorithm_dir.name == "splits":
                continue
            for category in _RESULT_CATEGORIES:
                result_files.extend(_csv_files_in_category(algorithm_dir, category))
    return result_files
