"""Randomization test view resolution and train/predict helpers."""

from __future__ import annotations

import os
import warnings
from typing import Any

from sklearn.base import TransformerMixin, clone

from ..datasets.dataset import DrugResponseDataset
from ..models.drp_model import DRPModel
from .training import load_features, train_and_predict_impl


def _resolve_cell_line_and_drug_views(
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    from drevalpy.components.data_loading.view_resolution import (
        cell_line_views_from_model_config,
        drug_views_from_model_config,
    )
    from drevalpy.components.tuning.public_flat import model_config_for_drp_model

    config = model_config_for_drp_model(model_class, hyperparameters)
    if config is not None:
        return (
            list(cell_line_views_from_model_config(config)),
            list(drug_views_from_model_config(config)),
        )
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


def _normalize_view_list(views: list[str] | str) -> list[str]:
    return [views] if isinstance(views, str) else list(views)


def _missing_randomization_views(
    view_list: list[str],
    cl_features,
    drug_features,
) -> list[str]:
    return [
        view
        for view in view_list
        if (cl_features is None or view not in cl_features.view_names)
        and (drug_features is None or view not in drug_features.view_names)
    ]


def _randomize_feature_views(
    view_list: list[str],
    cl_features,
    drug_features,
    randomization_type: str,
) -> tuple:
    cl_features_rand = cl_features.copy() if cl_features is not None else None
    drug_features_rand = drug_features.copy() if drug_features is not None else None
    for view in view_list:
        if cl_features_rand is not None and view in cl_features_rand.view_names:
            cl_features_rand.randomize_features(view, randomization_type=randomization_type)
        if drug_features_rand is not None and view in drug_features_rand.view_names:
            drug_features_rand.randomize_features(view, randomization_type=randomization_type)
    return cl_features_rand, drug_features_rand


def randomize_train_predict_impl(
    views: list[str] | str,
    test_name: str,
    randomization_type: str,
    randomization_test_file: str,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    model_checkpoint_dir: str = "TEMPORARY",
    response_transformation: TransformerMixin | None = None,
) -> None:
    """Randomize views, train once, and write predictions.

    :param views: Feature view or views to randomize.
    :param test_name: Label for the randomization test output.
    :param randomization_type: Randomization strategy (for example ``permutation``).
    :param randomization_test_file: Output path for predictions.
    :param model_class: Model class to train under randomized inputs.
    :param hyperparameters: Hyperparameters for model construction.
    :param path_data: Root directory for feature tables.
    :param train_dataset: Training split for the fold.
    :param test_dataset: Test split for the fold.
    :param early_stopping_dataset: Optional early-stopping data.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param response_transformation: Optional response transformer.
    """
    view_list = _normalize_view_list(views)
    trial_model = model_class(hyperparameters)
    cl_features, drug_features = load_features(trial_model, path_data, train_dataset)

    if cl_features is None and drug_features is None:
        warnings.warn(
            "Both cl_features and drug_features are None. Skipping randomization test.",
            stacklevel=2,
        )
        return

    missing = _missing_randomization_views(view_list, cl_features, drug_features)
    if missing:
        warnings.warn(
            f"Views {missing} not found in features. Skipping randomization test {test_name}.",
            stacklevel=2,
        )
        return

    cl_features_rand, drug_features_rand = _randomize_feature_views(
        view_list, cl_features, drug_features, randomization_type
    )
    trial_transform = None if response_transformation is None else clone(response_transformation)
    test_dataset_rand = train_and_predict_impl(
        model=trial_model,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=test_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=trial_transform,
        cl_features=cl_features_rand,
        drug_features=drug_features_rand,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    test_dataset_rand.to_csv(randomization_test_file)


def randomization_test_impl(
    randomization_test_views: dict[str, list[str]],
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    path_out: str,
    split_index: int,
    randomization_type: str = "permutation",
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Run randomization tests once per view configuration.

    :param randomization_test_views: Mapping from test names to feature views.
    :param model_class: Model class to train under randomized inputs.
    :param hyperparameters: Hyperparameters for model construction.
    :param path_data: Root directory for feature tables.
    :param train_dataset: Training split for the fold.
    :param test_dataset: Test split for the fold.
    :param early_stopping_dataset: Optional early-stopping data.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param randomization_type: Randomization strategy (for example ``permutation``).
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints.
    """
    for test_name, views in randomization_test_views.items():
        randomization_test_path = os.path.join(path_out, "randomization")
        os.makedirs(randomization_test_path, exist_ok=True)
        randomization_test_file = os.path.join(
            randomization_test_path,
            f"randomization_{test_name}_split_{split_index}.csv",
        )
        if os.path.isfile(randomization_test_file):
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
            path_data=path_data,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            early_stopping_dataset=early_stopping_dataset,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
