"""Randomization test view resolution and train/predict helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone
from upath import UPath as Path

from ..data.structures import EntityScope
from ..data.structures.mudataset import MuDataset
from ..models.drp_model import DRPModel
from .training import mu_train_and_predict


def _resolve_cell_line_and_drug_views(
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    from drevalpy.components.tuning.public_flat import model_config_for_drp_model

    config = model_config_for_drp_model(model_class, hyperparameters)
    if config is not None:
        return list(config.cell_line_views()), list(config.drug_views())
    probe = model_class(hyperparameters)
    return list(probe.cell_line_views), list(probe.drug_views)


def _append_svcc_views(randomization_test_views: dict[str, list[str]], cell_line_views: list[str]) -> None:
    for view in cell_line_views:
        randomization_test_views[f"SVCC_{view}"] = [v for v in cell_line_views if v != view]


def _append_svcd_views(randomization_test_views: dict[str, list[str]], drug_views: list[str]) -> None:
    for view in drug_views:
        randomization_test_views[f"SVCD_{view}"] = [v for v in drug_views if v != view]


def _append_svrc_views(randomization_test_views: dict[str, list[str]], cell_line_views: list[str]) -> None:
    for view in cell_line_views:
        randomization_test_views[f"SVRC_{view}"] = [view]


def _append_svrd_views(randomization_test_views: dict[str, list[str]], drug_views: list[str]) -> None:
    for view in drug_views:
        randomization_test_views[f"SVRD_{view}"] = [view]


def build_randomization_test_views(
    model_class: type[DRPModel],
    randomization_mode: list[str],
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Build mapping of randomization test name to views randomized together.

    :param model_class: Model class whose featurizers define available views.
    :param randomization_mode: Requested randomization modes (for example ``SVCC``).
    :param hyperparameters: Model hyperparameters used to resolve view names.

    :returns: Mapping from test names to feature-view lists.
    """
    cell_line_views, drug_views = _resolve_cell_line_and_drug_views(model_class, hyperparameters)
    randomization_test_views: dict[str, list[str]] = {}
    if "SVCC" in randomization_mode:
        _append_svcc_views(randomization_test_views, cell_line_views)
    if "SVCD" in randomization_mode:
        _append_svcd_views(randomization_test_views, drug_views)
    if "SVRC" in randomization_mode:
        _append_svrc_views(randomization_test_views, cell_line_views)
    if "SVRD" in randomization_mode:
        _append_svrd_views(randomization_test_views, drug_views)
    return randomization_test_views


def _available_views(mudataset: MuDataset) -> set[str]:
    """Gather all view names available in the MuDataset."""
    views: set[str] = set(mudataset.mdata.mod.keys()) - {"response"}
    if mudataset.response.varm:
        views.update(mudataset.response.varm.keys())
    if "pathway_features" in (mudataset.response.obsm or {}):
        views.add("pathway_features")
    return views


def _missing_randomization_views(view_list: list[str], mudataset: MuDataset) -> list[str]:
    """Return views from view_list not present in the MuDataset."""
    available = _available_views(mudataset)
    return [v for v in view_list if v not in available]


def _write_randomization_predictions(
    prediction_file: Path,
    mudataset: MuDataset,
    test_scope: EntityScope,
    predictions: np.ndarray,
) -> None:
    """Write randomization test prediction CSV."""
    cl_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids
    response_matrix = mudataset.response_matrix

    cl_idx = test_scope.cell_lines
    dr_idx = test_scope.drugs

    rows: dict[str, Any] = {"cell_line_ids": cl_ids[cl_idx]}
    if dr_idx is not None:
        rows["drug_ids"] = drug_ids[dr_idx]
        rows["response"] = response_matrix[cl_idx, dr_idx]
    else:
        rows["drug_ids"] = np.full(len(cl_idx), "all", dtype=object)
        rows["response"] = np.nanmean(response_matrix[cl_idx, :], axis=1)
    rows["predictions"] = predictions

    df = pd.DataFrame(rows)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prediction_file, index=False)


def randomize_train_predict_impl(
    views: list[str] | str,
    test_name: str,
    randomization_type: str,
    randomization_test_file: str | Path,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None = None,
    model_checkpoint_dir: str | Path | None = None,
    response_transformation: TransformerMixin | None = None,
) -> None:
    """Randomize views, train once, and write predictions.

    :param views: Feature view or views to randomize.
    :param test_name: Label for the randomization test output.
    :param randomization_type: Randomization strategy (for example ``permutation``).
    :param randomization_test_file: Output path for predictions.
    :param model_class: Model class to train under randomized inputs.
    :param hyperparameters: Hyperparameters for model construction.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :param response_transformation: Optional response transformer.
    """
    view_list = [views] if isinstance(views, str) else list(views)

    missing = _missing_randomization_views(view_list, mudataset)
    if missing:
        warnings.warn(
            f"Views {missing} not found in MuDataset. Skipping randomization test {test_name}.",
            stacklevel=2,
        )
        return

    randomized_mudataset = mudataset.with_randomized_views(
        views=view_list,
        randomization_type=randomization_type,
    )

    trial_model = model_class(hyperparameters)
    trial_transform = None if response_transformation is None else clone(response_transformation)

    predictions = mu_train_and_predict(
        model=trial_model,
        mudataset=randomized_mudataset,
        train_scope=train_scope,
        test_scope=test_scope,
        early_stopping_scope=early_stopping_scope,
        response_transformation=trial_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )

    _write_randomization_predictions(
        Path(randomization_test_file),
        mudataset,
        test_scope,
        predictions,
    )


def randomization_test_impl(
    randomization_test_views: dict[str, list[str]],
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    path_out: str | Path,
    split_index: int,
    randomization_type: str = "permutation",
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Run randomization tests once per view configuration.

    :param randomization_test_views: Mapping from test names to feature views.
    :param model_class: Model class to train under randomized inputs.
    :param hyperparameters: Hyperparameters for model construction.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param randomization_type: Randomization strategy (for example ``permutation``).
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    """
    for test_name, views in randomization_test_views.items():
        randomization_test_path = Path(path_out) / "randomization"
        randomization_test_path.mkdir(parents=True, exist_ok=True)
        randomization_test_file = randomization_test_path / f"randomization_{test_name}_split_{split_index}.csv"
        if randomization_test_file.is_file():
            print(f"Randomization test {test_name} already exists. Skipping.")
            continue
        print(f"Randomizing views {views} for randomization test {test_name} ...")
        randomize_train_predict_impl(
            views=views,
            test_name=test_name,
            randomization_type=randomization_type,
            randomization_test_file=randomization_test_file,
            model_class=model_class,
            hyperparameters=hyperparameters,
            mudataset=mudataset,
            train_scope=train_scope,
            test_scope=test_scope,
            early_stopping_scope=early_stopping_scope,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
