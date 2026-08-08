"""Consolidate per-drug prediction artifacts for single-drug models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..models._model_lookup import get_model_class, is_single_drug_model_name
from ..models.drp_model import DRPModel
from .randomization import build_randomization_test_views

_CWD = Path()


def _ensure_consolidated_dirs(
    out_path: Path,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None,
    n_trials_robustness: int,
) -> None:
    (out_path / "predictions").mkdir(parents=True, exist_ok=True)
    if cross_study_datasets:
        (out_path / "cross_study").mkdir(parents=True, exist_ok=True)
    if randomization_mode:
        (out_path / "randomization").mkdir(parents=True, exist_ok=True)
    if n_trials_robustness:
        (out_path / "robustness").mkdir(parents=True, exist_ok=True)


def _list_drug_ids(model_path: Path) -> list[str]:
    drugs_dir = model_path / "drugs"
    return [entry.name for entry in drugs_dir.iterdir() if entry.is_dir()]


def _read_main_prediction(single_drug_prediction_path: Path, split: int) -> pd.DataFrame:
    path = single_drug_prediction_path / "predictions" / f"predictions_split_{split}.csv"
    return pd.read_csv(path, index_col=0)


def _accumulate_cross_study(
    predictions: dict[str, Any],
    single_drug_prediction_path: Path,
    cross_study_datasets: list[str],
    split: int,
) -> None:
    for cross_study_dataset in cross_study_datasets:
        cross_study_prediction_path = single_drug_prediction_path / "cross_study"
        filename = f"cross_study_{cross_study_dataset}_split_{split}.csv"
        if cross_study_dataset not in predictions["cross_study"]:
            predictions["cross_study"][cross_study_dataset] = []
        file_path = cross_study_prediction_path / filename
        if not file_path.exists():
            continue
        predictions["cross_study"][cross_study_dataset].append(pd.read_csv(file_path, index_col=0))


def _accumulate_robustness(
    predictions: dict[str, Any],
    single_drug_prediction_path: Path,
    n_trials_robustness: int,
    split: int,
) -> None:
    for trial in range(n_trials_robustness):
        robustness_path = single_drug_prediction_path / "robustness"
        filename = f"robustness_{trial + 1}_split_{split}.csv"
        if trial not in predictions["robustness"]:
            predictions["robustness"][trial] = []
        predictions["robustness"][trial].append(pd.read_csv(robustness_path / filename, index_col=0))


def _accumulate_randomization(
    predictions: dict[str, Any],
    single_drug_prediction_path: Path,
    model_class: type[DRPModel],
    randomization_mode: list[str] | None,
    split: int,
) -> None:
    if randomization_mode is None:
        return
    randomization_test_views = build_randomization_test_views(
        model_class=model_class,
        randomization_mode=randomization_mode,
    )
    randomization_path = single_drug_prediction_path / "randomization"
    for view in randomization_test_views:
        filename = f"randomization_{view}_split_{split}.csv"
        if view not in predictions["randomization"]:
            predictions["randomization"][view] = []
        predictions["randomization"][view].append(pd.read_csv(randomization_path / filename, index_col=0))


def _collect_split_predictions(
    model_path: Path,
    model_class: type[DRPModel],
    split: int,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None,
    n_trials_robustness: int,
) -> dict[str, Any]:
    predictions: dict[str, Any] = {
        "main": [],
        "cross_study": {},
        "robustness": {},
        "randomization": {},
    }
    for drug in _list_drug_ids(model_path):
        single_drug_path = model_path / "drugs" / drug
        predictions["main"].append(_read_main_prediction(single_drug_path, split))
        _accumulate_cross_study(predictions, single_drug_path, cross_study_datasets, split)
        _accumulate_robustness(predictions, single_drug_path, n_trials_robustness, split)
        _accumulate_randomization(predictions, single_drug_path, model_class, randomization_mode, split)
    return predictions


def _write_consolidated_split(
    predictions: dict[str, Any],
    out_path: Path,
    split: int,
) -> None:
    pd.concat(predictions["main"], axis=0).to_csv(out_path / "predictions" / f"predictions_split_{split}.csv")
    for dataset_name, dataset_predictions in predictions["cross_study"].items():
        if not dataset_predictions:
            continue
        pd.concat(dataset_predictions, axis=0).to_csv(
            out_path / "cross_study" / f"cross_study_{dataset_name}_split_{split}.csv"
        )
    for trial, trial_predictions in predictions["robustness"].items():
        pd.concat(trial_predictions, axis=0).to_csv(
            out_path / "robustness" / f"robustness_{trial + 1}_split_{split}.csv"
        )
    for view, view_predictions in predictions["randomization"].items():
        pd.concat(view_predictions, axis=0).to_csv(
            out_path / "randomization" / f"randomization_{view}_split_{split}.csv"
        )


def consolidate_single_drug_model_predictions_impl(
    models: list[type[DRPModel]],
    n_cv_splits: int,
    results_path: str | Path,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None = None,
    n_trials_robustness: int = 0,
    out_path: str | Path = _CWD,
) -> None:
    """Consolidate single-drug per-drug CSVs into model-level files.

    :param models: Model classes whose outputs should be consolidated.
    :param n_cv_splits: Number of CV folds written during the experiment.
    :param results_path: Experiment result directory to read from.
    :param cross_study_datasets: Names of cross-study datasets to include.
    :param randomization_mode: Randomization views to consolidate, if any.
    :param n_trials_robustness: Number of robustness trials to consolidate.
    :param out_path: Output directory; defaults to the current working directory.
    """
    results_root = Path(results_path)
    out_root = Path(out_path)
    for model in models:
        if not is_single_drug_model_name(model.get_model_name()):
            continue
        model_class = get_model_class(model.get_model_name())
        model_path = results_root / model.get_model_name()
        model_out = out_root / model.get_model_name()
        _ensure_consolidated_dirs(model_out, cross_study_datasets, randomization_mode, n_trials_robustness)

        for split in range(n_cv_splits):
            predictions = _collect_split_predictions(
                model_path,
                model_class,
                split,
                cross_study_datasets,
                randomization_mode,
                n_trials_robustness,
            )
            _write_consolidated_split(predictions, model_out, split)
