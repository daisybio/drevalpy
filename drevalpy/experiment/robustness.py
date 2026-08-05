"""Robustness testing helpers for experiment workflows."""

from __future__ import annotations

import os

from sklearn.base import TransformerMixin, clone

from ..datasets.dataset import DrugResponseDataset
from ..models.drp_model import DRPModel
from .training import train_and_predict_impl


def robustness_train_predict_impl(
    trial: int,
    trial_file: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    model_class: type[DRPModel],
    hyperparameters: dict,
    path_data: str,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Train and predict for one robustness trial.

    :param trial: Trial index within the robustness test.
    :param trial_file: Output path for predictions.
    :param train_dataset: Training split for the fold.
    :param test_dataset: Test split for the fold.
    :param early_stopping_dataset: Optional early-stopping data.
    :param model_class: Model class to train on perturbed data.
    :param hyperparameters: Hyperparameters for model construction.
    :param path_data: Root directory for feature tables.
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints.
    """
    train_trial = train_dataset.shuffled(random_state=trial)
    test_trial = test_dataset.shuffled(random_state=trial)
    es_trial = early_stopping_dataset.shuffled(random_state=trial) if early_stopping_dataset is not None else None
    trial_model = model_class(hyperparameters)
    trial_transform = None if response_transformation is None else clone(response_transformation)
    predicted = train_and_predict_impl(
        model=trial_model,
        path_data=path_data,
        train_dataset=train_trial,
        prediction_dataset=test_trial,
        early_stopping_dataset=es_trial,
        response_transformation=trial_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    predicted.to_csv(trial_file)


def robustness_test_impl(
    n_trials: int,
    model_class: type[DRPModel],
    hyperparameters: dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    path_out: str,
    split_index: int,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Run robustness tests with varying shuffle seeds.

    :param n_trials: Number of robustness trials to run.
    :param model_class: Model class to retrain on perturbed data.
    :param hyperparameters: Hyperparameters for model construction.
    :param path_data: Root directory for feature tables.
    :param train_dataset: Training split for the fold.
    :param test_dataset: Test split for the fold.
    :param early_stopping_dataset: Optional early-stopping data.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints.
    """
    robustness_test_path = os.path.join(path_out, "robustness")
    os.makedirs(robustness_test_path, exist_ok=True)
    for trial in range(n_trials):
        print(f"Running robustness test trial {trial + 1}/{n_trials}")
        trial_file = os.path.join(robustness_test_path, f"robustness_{trial + 1}_split_{split_index}.csv")
        if os.path.isfile(trial_file):
            continue
        robustness_train_predict_impl(
            trial=trial,
            trial_file=trial_file,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            early_stopping_dataset=early_stopping_dataset,
            model_class=model_class,
            hyperparameters=hyperparameters,
            path_data=path_data,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
