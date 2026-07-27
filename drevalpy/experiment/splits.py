"""CV split preparation for experiment runs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..datasets.dataset import DrugResponseDataset
from ..datasets.splits import ExternalSplitCreator, create_and_record_splits


def prepare_response_splits_impl(
    response_data: DrugResponseDataset,
    *,
    split_path: str,
    result_path: str,
    split_label: str,
    test_mode: str,
    n_cv_splits: int,
    overwrite: bool,
    result_folder_exists: bool,
    custom_splitter: ExternalSplitCreator | str | Path | None = None,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    split_early_stopping: bool = True,
) -> int:
    """Create, load, or reuse CV splits for an experiment run."""
    if result_folder_exists and overwrite:
        print(f"Overwriting existing results at {result_path}")
        shutil.rmtree(result_path)

    if result_folder_exists and os.path.exists(split_path) and not overwrite:
        print(f"Loading existing cv splits from {split_path}")
        response_data.load_splits(path=split_path)
    else:
        print(f"Creating cv splits at {split_path}")
        os.makedirs(result_path, exist_ok=True)
        create_and_record_splits(
            response_data,
            split_path=split_path,
            split_label=split_label,
            external_splitter=custom_splitter,
            test_mode=test_mode,
            n_cv_splits=n_cv_splits,
            validation_ratio=validation_ratio,
            random_state=random_state,
            split_early_stopping=split_early_stopping,
        )
        response_data.save_splits(path=split_path)

    return len(response_data.cv_splits)
