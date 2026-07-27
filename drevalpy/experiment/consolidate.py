"""Consolidate per-drug prediction artifacts for single-drug models."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from ..models._model_lookup import get_model_class, is_single_drug_model_name
from ..models.drp_model import DRPModel
from .randomization import build_randomization_test_views


def _ensure_consolidated_dirs(
    out_path: str,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None,
    n_trials_robustness: int,
) -> None:
    os.makedirs(os.path.join(out_path, "predictions"), exist_ok=True)
    if cross_study_datasets:
        os.makedirs(os.path.join(out_path, "cross_study"), exist_ok=True)
    if randomization_mode:
        os.makedirs(os.path.join(out_path, "randomization"), exist_ok=True)
    if n_trials_robustness:
        os.makedirs(os.path.join(out_path, "robustness"), exist_ok=True)


def _list_drug_ids(model_path: str) -> list[str]:
    drugs_dir = os.path.join(model_path, "drugs")
    return [d for d in os.listdir(drugs_dir) if os.path.isdir(os.path.join(drugs_dir, d))]


def _read_main_prediction(single_drug_prediction_path: str, split: int) -> pd.DataFrame:
    path = os.path.join(single_drug_prediction_path, "predictions", f"predictions_split_{split}.csv")
    return pd.read_csv(path, index_col=0)


def _accumulate_cross_study(
    predictions: dict[str, Any],
    single_drug_prediction_path: str,
    cross_study_datasets: list[str],
    split: int,
) -> None:
    for cross_study_dataset in cross_study_datasets:
        cross_study_prediction_path = os.path.join(single_drug_prediction_path, "cross_study")
        filename = f"cross_study_{cross_study_dataset}_split_{split}.csv"
        if cross_study_dataset not in predictions["cross_study"]:
            predictions["cross_study"][cross_study_dataset] = []
        predictions["cross_study"][cross_study_dataset].append(
            pd.read_csv(os.path.join(cross_study_prediction_path, filename), index_col=0)
        )


def _accumulate_robustness(
    predictions: dict[str, Any],
    single_drug_prediction_path: str,
    n_trials_robustness: int,
    split: int,
) -> None:
    for trial in range(n_trials_robustness):
        robustness_path = os.path.join(single_drug_prediction_path, "robustness")
        filename = f"robustness_{trial + 1}_split_{split}.csv"
        if trial not in predictions["robustness"]:
            predictions["robustness"][trial] = []
        predictions["robustness"][trial].append(pd.read_csv(os.path.join(robustness_path, filename), index_col=0))


def _accumulate_randomization(
    predictions: dict[str, Any],
    single_drug_prediction_path: str,
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
    randomization_path = os.path.join(single_drug_prediction_path, "randomization")
    for view in randomization_test_views:
        filename = f"randomization_{view}_split_{split}.csv"
        if view not in predictions["randomization"]:
            predictions["randomization"][view] = []
        predictions["randomization"][view].append(pd.read_csv(os.path.join(randomization_path, filename), index_col=0))


def _collect_split_predictions(
    model_path: str,
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
        single_drug_path = os.path.join(model_path, "drugs", drug)
        predictions["main"].append(_read_main_prediction(single_drug_path, split))
        _accumulate_cross_study(predictions, single_drug_path, cross_study_datasets, split)
        _accumulate_robustness(predictions, single_drug_path, n_trials_robustness, split)
        _accumulate_randomization(predictions, single_drug_path, model_class, randomization_mode, split)
    return predictions


def _write_consolidated_split(
    predictions: dict[str, Any],
    out_path: str,
    split: int,
) -> None:
    pd.concat(predictions["main"], axis=0).to_csv(
        os.path.join(out_path, "predictions", f"predictions_split_{split}.csv")
    )
    for dataset_name, dataset_predictions in predictions["cross_study"].items():
        pd.concat(dataset_predictions, axis=0).to_csv(
            os.path.join(out_path, "cross_study", f"cross_study_{dataset_name}_split_{split}.csv")
        )
    for trial, trial_predictions in predictions["robustness"].items():
        pd.concat(trial_predictions, axis=0).to_csv(
            os.path.join(out_path, "robustness", f"robustness_{trial + 1}_split_{split}.csv")
        )
    for view, view_predictions in predictions["randomization"].items():
        pd.concat(view_predictions, axis=0).to_csv(
            os.path.join(out_path, "randomization", f"randomization_{view}_split_{split}.csv")
        )


def consolidate_single_drug_model_predictions_impl(
    models: list[type[DRPModel]],
    n_cv_splits: int,
    results_path: str,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None = None,
    n_trials_robustness: int = 0,
    out_path: str = "",
) -> None:
    """Consolidate single-drug per-drug CSVs into model-level files."""
    for model in models:
        if not is_single_drug_model_name(model.get_model_name()):
            continue
        model_class = get_model_class(model.get_model_name())
        model_path = os.path.join(results_path, model.get_model_name())
        model_out = os.path.join(out_path, model.get_model_name())
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
