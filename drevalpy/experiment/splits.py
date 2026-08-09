"""CV split preparation for experiment runs."""

from __future__ import annotations

import json
import shutil

from upath import UPath as Path

from ..data.splitters import Splitter, splitter_registry
from ..data.structures import MuDataset, SplitMasks


def prepare_splits(
    mudataset: MuDataset,
    *,
    split_path: str | Path,
    result_path: str | Path,
    test_mode: str | Splitter,
    n_cv_splits: int,
    overwrite: bool,
    result_folder_exists: bool,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Create, load, or reuse CV splits for an experiment run.

    :param mudataset: MuDataset to split.
    :param split_path: Directory for split manifest and fold files.
    :param result_path: Experiment result directory.
    :param test_mode: Split mode ("LPO", "LCO", "LDO", "LTO").
    :param n_cv_splits: Requested number of folds.
    :param overwrite: Rebuild splits even when cached splits exist.
    :param result_folder_exists: Whether ``result_path`` already exists.
    :param validation_ratio: Fraction of training data held out for validation.
    :param random_state: Random seed for splitting.

    :returns: List of SplitMasks, one per fold.
    """
    splits_dir = Path(split_path)
    results_dir = Path(result_path)
    manifest_file = splits_dir / "splits_manifest.json"

    if result_folder_exists and overwrite:
        print(f"Overwriting existing results at {results_dir}")
        shutil.rmtree(results_dir)

    if result_folder_exists and manifest_file.is_file() and not overwrite:
        print(f"Loading existing cv splits from {splits_dir}")
        return _load_splits_from_dir(splits_dir)

    print(f"Creating cv splits at {splits_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    splitter = splitter_registry.resolve(test_mode)
    folds = splitter.split(
        mudataset,
        n_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )

    _save_splits_to_dir(folds, splits_dir, test_mode=test_mode)
    return folds


def _save_splits_to_dir(folds: list[SplitMasks], splits_dir: Path, *, test_mode: str) -> None:
    """Persist split masks as .npz files plus a JSON manifest."""
    manifest = {
        "test_mode": test_mode,
        "n_folds": len(folds),
    }
    for i, fold in enumerate(folds):
        fold.save(splits_dir / f"fold_{i}.npz")

    with open(splits_dir / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def _load_splits_from_dir(splits_dir: Path) -> list[SplitMasks]:
    """Load saved split masks from .npz files."""
    with open(splits_dir / "splits_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    folds: list[SplitMasks] = []
    for i in range(manifest["n_folds"]):
        folds.append(SplitMasks.load(splits_dir / f"fold_{i}.npz"))
    return folds
