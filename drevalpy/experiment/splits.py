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
    """Create, load, or reuse CV splits for an experiment run.

    :param response_data: Dataset that receives ``cv_splits`` in place.
    :param split_path: Directory for split manifest and fold files.
    :param result_path: Experiment result directory.
    :param split_label: Label stored in the split manifest.
    :param test_mode: Builtin split mode or label for external splits.
    :param n_cv_splits: Requested number of folds.
    :param overwrite: Rebuild splits even when a manifest already exists.
    :param result_folder_exists: Whether ``result_path`` already exists.
    :param custom_splitter: External split creator or manifest path.
    :param validation_ratio: Fraction of training data held out for validation.
    :param random_state: Random seed for builtin splitters.
    :param split_early_stopping: Whether to create early-stopping folds.

    :returns: Actual number of CV splits attached to *response_data*.
    """
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
