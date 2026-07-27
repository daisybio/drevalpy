"""Main module for running the drug response prediction experiment."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from sklearn.base import TransformerMixin, clone

from drevalpy.components.tuning.hpo import hpam_tune

from ..datasets.dataset import DrugResponseDataset, FeatureDataset
from ..datasets.splits import ExternalSplitCreator
from ..models.drp_model import DRPModel
from ..pipeline_function import pipeline_function
from . import fold as _fold_module
from .consolidate import consolidate_single_drug_model_predictions_impl
from .cross_study import cross_study_prediction_impl
from .final_model import train_final_model_impl
from .fold import make_train_val_split_impl
from .model_paths import generate_data_saving_path as _generate_data_saving_path
from .model_paths import get_model_name_and_drug_id as _get_model_name_and_drug_id
from .model_paths import make_model_list as _make_model_list
from .randomization import build_randomization_test_views as _build_randomization_test_views
from .randomization import (
    randomization_test_impl,
    randomize_train_predict_impl,
)
from .robustness import robustness_test_impl, robustness_train_predict_impl
from .run import drug_response_experiment_impl
from .seed import seed_everything as seed_everything  # noqa: F401 — public re-export
from .splits import prepare_response_splits_impl
from .training import train_and_predict_impl

# Public compatibility re-export (callers may import from ``drevalpy.experiment``).
get_datasets_from_cv_split = _fold_module.get_datasets_from_cv_split

if importlib.util.find_spec("ray"):
    import ray
else:
    ray = None  # type: ignore[assignment]


@pipeline_function
def prepare_response_splits(
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
    return prepare_response_splits_impl(
        response_data,
        split_path=split_path,
        result_path=result_path,
        split_label=split_label,
        test_mode=test_mode,
        n_cv_splits=n_cv_splits,
        overwrite=overwrite,
        result_folder_exists=result_folder_exists,
        custom_splitter=custom_splitter,
        validation_ratio=validation_ratio,
        random_state=random_state,
        split_early_stopping=split_early_stopping,
    )


def drug_response_experiment(
    models: list[type[DRPModel]],
    response_data: DrugResponseDataset,
    baselines: list[type[DRPModel]] | None = None,
    response_transformation: TransformerMixin | None = None,
    run_id: str = "",
    test_mode: str = "LPO",
    hpam_optimization_metric: str = "RMSE",
    n_cv_splits: int = 5,
    multiprocessing: bool = False,
    randomization_mode: list[str] | None = None,
    randomization_type: str = "permutation",
    cross_study_datasets: list[DrugResponseDataset] | None = None,
    n_trials_robustness: int = 0,
    path_out: str = "results/",
    overwrite: bool = False,
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
    hyperparameter_tuning=True,
    final_model_on_full_data: bool = False,
    wandb_project: str | None = None,
    custom_splitter: ExternalSplitCreator | str | Path | None = None,
    custom_split_name: str | None = None,
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    hpo_resources_per_trial: dict[str, float] | None = None,
) -> None:
    """Run the drug response prediction experiment. Save results to disc."""
    drug_response_experiment_impl(
        models=models,
        response_data=response_data,
        baselines=baselines,
        response_transformation=response_transformation,
        run_id=run_id,
        test_mode=test_mode,
        hpam_optimization_metric=hpam_optimization_metric,
        n_cv_splits=n_cv_splits,
        multiprocessing=multiprocessing,
        randomization_mode=randomization_mode,
        randomization_type=randomization_type,
        cross_study_datasets=cross_study_datasets,
        n_trials_robustness=n_trials_robustness,
        path_out=path_out,
        overwrite=overwrite,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        hyperparameter_tuning=hyperparameter_tuning,
        final_model_on_full_data=final_model_on_full_data,
        wandb_project=wandb_project,
        custom_splitter=custom_splitter,
        custom_split_name=custom_split_name,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        hpo_resources_per_trial=hpo_resources_per_trial,
    )


@pipeline_function
def consolidate_single_drug_model_predictions(
    models: list[type[DRPModel]],
    n_cv_splits: int,
    results_path: str,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None = None,
    n_trials_robustness: int = 0,
    out_path: str = "",
) -> None:
    """Consolidate single drug model predictions into a single file."""
    consolidate_single_drug_model_predictions_impl(
        models=models,
        n_cv_splits=n_cv_splits,
        results_path=results_path,
        cross_study_datasets=cross_study_datasets,
        randomization_mode=randomization_mode,
        n_trials_robustness=n_trials_robustness,
        out_path=out_path,
    )


@pipeline_function
def cross_study_prediction(
    dataset: DrugResponseDataset,
    model: DRPModel,
    test_mode: str,
    train_dataset: DrugResponseDataset,
    path_data: str,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    path_out: str,
    split_index: int,
    single_drug_id: str | None = None,
) -> None:
    """Run cross-study prediction to assess model generalizability."""
    cross_study_prediction_impl(
        dataset=dataset,
        model=model,
        test_mode=test_mode,
        train_dataset=train_dataset,
        path_data=path_data,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
        path_out=path_out,
        split_index=split_index,
        single_drug_id=single_drug_id,
    )


@pipeline_function
def get_randomization_test_views(
    model_class: type[DRPModel],
    randomization_mode: list[str],
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Get the views to use for the randomization tests."""
    return _build_randomization_test_views(model_class, randomization_mode, hyperparameters)


def randomization_test(
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
    """Run randomization tests."""
    randomization_test_impl(
        randomization_test_views=randomization_test_views,
        model_class=model_class,
        hyperparameters=hyperparameters,
        path_data=path_data,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        early_stopping_dataset=early_stopping_dataset,
        path_out=path_out,
        split_index=split_index,
        randomization_type=randomization_type,
        response_transformation=response_transformation,
        model_checkpoint_dir=model_checkpoint_dir,
    )


@pipeline_function
def randomize_train_predict(
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
    """Randomize feature views, then train and predict once."""
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
        model_checkpoint_dir=model_checkpoint_dir,
        response_transformation=response_transformation,
    )


def robustness_test(
    n_trials: int,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    path_out: str,
    split_index: int,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
):
    """Run robustness tests for the given model and dataset."""
    robustness_test_impl(
        n_trials=n_trials,
        model_class=model_class,
        hyperparameters=hyperparameters,
        path_data=path_data,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        early_stopping_dataset=early_stopping_dataset,
        path_out=path_out,
        split_index=split_index,
        response_transformation=response_transformation,
        model_checkpoint_dir=model_checkpoint_dir,
    )


@pipeline_function
def robustness_train_predict(
    trial: int,
    trial_file: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    path_data: str,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Train and predict for the robustness test with a freshly constructed model."""
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


def split_early_stopping(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """Split the validation dataset into validation and early stopping datasets."""
    validation_dataset = validation_dataset.shuffled(random_state=42)
    cv_v = validation_dataset.split_dataset(
        n_cv_splits=4,
        mode=test_mode,
        split_validation=False,
        random_state=42,
    )
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


@pipeline_function
def train_and_predict(
    model: DRPModel,
    path_data: str,
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None = None,
    response_transformation: TransformerMixin | None = None,
    cl_features: FeatureDataset | None = None,
    drug_features: FeatureDataset | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> DrugResponseDataset:
    """Train the model and predict the response for the prediction dataset."""
    return train_and_predict_impl(
        model=model,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=prediction_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
        cl_features=cl_features,
        drug_features=drug_features,
        model_checkpoint_dir=model_checkpoint_dir,
    )


def train_and_evaluate(
    model: DRPModel,
    path_data: str,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None = None,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    model_checkpoint_dir: str = "TEMPORARY",
) -> dict[str, float]:
    """Train and evaluate the model."""
    trial_transform = None if response_transformation is None else clone(response_transformation)
    validation_dataset = train_and_predict(
        model=model,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=trial_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )

    additional_metrics = None
    if metric not in ["R^2", "Pearson"]:
        additional_metrics = [metric]
    return model.compute_and_log_final_metrics(
        validation_dataset,
        additional_metrics=additional_metrics,
        prefix="val_",
    )


@pipeline_function
def make_model_list(models: list[type[DRPModel]], response_data: DrugResponseDataset) -> dict[str, str]:
    """Make a list of models to evaluate."""
    return _make_model_list(models, response_data)


@pipeline_function
def get_model_name_and_drug_id(model_name: str) -> tuple[str, str | None]:
    """Get the model name and drug id from the model name."""
    return _get_model_name_and_drug_id(model_name)


@pipeline_function
def generate_data_saving_path(model_name, drug_id, result_path, suffix) -> str:
    """Generate a path to save data to."""
    return _generate_data_saving_path(model_name, drug_id, result_path, suffix)


def train_final_model(
    model_class: type[DRPModel],
    full_dataset: DrugResponseDataset,
    response_transformation: TransformerMixin,
    path_data: str,
    model_checkpoint_dir: str,
    metric: str,
    final_model_path: str,
    test_mode: str = "LCO",
    val_ratio: float = 0.1,
    hyperparameter_tuning: bool = True,
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    hpo_resources_per_trial: dict[str, float] | None = None,
    hpo_storage_path: str | None = None,
) -> None:
    """Final production model training."""
    train_final_model_impl(
        model_class=model_class,
        full_dataset=full_dataset,
        response_transformation=response_transformation,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        metric=metric,
        final_model_path=final_model_path,
        test_mode=test_mode,
        val_ratio=val_ratio,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        hpo_resources_per_trial=hpo_resources_per_trial,
        hpo_storage_path=hpo_storage_path,
    )


@pipeline_function
def make_train_val_split(
    dataset: DrugResponseDataset,
    test_mode: str,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """Split a dataset into train and validation sets."""
    return make_train_val_split_impl(dataset, test_mode, val_ratio, random_state)


# Ray Tune entry points and tests import ``hpam_tune`` from ``drevalpy.experiment``.
hpam_tune = hpam_tune
