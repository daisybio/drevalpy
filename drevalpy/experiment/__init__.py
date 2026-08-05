"""Cross-validation loops and experiment orchestration entry points."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from sklearn.base import TransformerMixin, clone

from drevalpy.components.tuning.hpo import hpam_tune

from ..datasets.dataset import DrugResponseDataset, FeatureDataset
from ..datasets.splits import ExternalSplitCreator
from ..models.drp_model import DRPModel
from ..utils._pipeline_function import pipeline_function
from . import fold as _fold_module
from .consolidate import consolidate_single_drug_model_predictions_impl
from .cross_study import cross_study_prediction_impl
from .final_model import train_final_model_impl
from .fold import make_train_val_split_impl
from .model_paths import generate_data_saving_path as _generate_data_saving_path
from .model_paths import generate_final_model_checkpoint_path as _generate_final_model_checkpoint_path
from .model_paths import get_model_name_and_drug_id as _get_model_name_and_drug_id
from .model_paths import make_model_list as _make_model_list
from .randomization import build_randomization_test_views as _build_randomization_test_views
from .randomization import (
    randomization_test_impl,
    randomize_train_predict_impl,
)
from .robustness import robustness_test_impl, robustness_train_predict_impl
from .run import drug_response_experiment_impl
from .seed import seed_everything
from .splits import prepare_response_splits_impl
from .training import train_and_predict_impl

# Public compatibility re-export (callers may import from ``drevalpy.experiment``).
get_datasets_from_cv_split = _fold_module.get_datasets_from_cv_split

ray: ModuleType | None
if importlib.util.find_spec("ray") is not None:
    ray = importlib.import_module("ray")
else:
    ray = None

__all__ = [
    "consolidate_single_drug_model_predictions",
    "cross_study_prediction",
    "drug_response_experiment",
    "generate_data_saving_path",
    "generate_final_model_checkpoint_path",
    "get_datasets_from_cv_split",
    "get_model_name_and_drug_id",
    "get_randomization_test_views",
    "hpam_tune",
    "make_model_list",
    "make_train_val_split",
    "prepare_response_splits",
    "randomization_test",
    "randomize_train_predict",
    "ray",
    "robustness_test",
    "robustness_train_predict",
    "seed_everything",
    "split_early_stopping",
    "train_and_evaluate",
    "train_and_predict",
    "train_final_model",
]


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
    """Run the drug response prediction experiment and save results to disk.

    Trains each model across CV folds (with optional HPO), writes predictions
    and hyperparameters under ``path_out``, and optionally runs randomization,
    robustness, cross-study, and final-model workflows.

    :param models: ``DRPModel`` subclasses to evaluate (from ``construct_model``).
    :param response_data: Training/validation response table with CV splits attached.
    :param baselines: Optional baseline models; ``NaiveMeanEffectsPredictor`` is added by default when omitted.
    :param response_transformation: Optional sklearn transformer applied to responses.
    :param run_id: Subfolder name under ``path_out`` for this run.
    :param test_mode: Split mode (``"LPO"``, ``"LCO"``, ``"LDO"``, or custom).
    :param hpam_optimization_metric: Metric optimized during HPO (for example ``"RMSE"``).
    :param n_cv_splits: Number of cross-validation folds.
    :param multiprocessing: Deprecated; routes through Ray Tune when ``True``.
    :param randomization_mode: Feature views to permute for randomization tests.
    :param randomization_type: Permutation strategy for randomization tests.
    :param cross_study_datasets: Additional datasets for cross-study evaluation.
    :param n_trials_robustness: Number of robustness-test resampling trials.
    :param path_out: Root directory for experiment outputs.
    :param overwrite: Recompute splits and predictions even when artifacts exist.
    :param path_data: Root directory for feature tables.
    :param model_checkpoint_dir: Directory for per-fold model checkpoints.
    :param hyperparameter_tuning: Whether to run HPO before final fold training.
    :param final_model_on_full_data: Train a production model on all data after CV.
    :param wandb_project: Optional Weights & Biases project name.
    :param custom_splitter: External split creator or path to split manifest.
    :param custom_split_name: Label for custom splits in output paths.
    :param hpo_num_samples: Number of HPO trials per fold.
    :param hpo_random_state: Random seed for HPO search.
    :param hpo_resources_per_trial: Optional Ray resource limits per HPO trial.
    """
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
    """Consolidate per-fold single-drug predictions into summary files.

    :param models: Model classes whose outputs should be consolidated.
    :param n_cv_splits: Number of CV folds written during the experiment.
    :param results_path: Experiment result directory to read from.
    :param cross_study_datasets: Names of cross-study datasets to include.
    :param randomization_mode: Randomization views to consolidate, if any.
    :param n_trials_robustness: Number of robustness trials to consolidate.
    :param out_path: Output directory; defaults to *results_path* when empty.
    """
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
    """Run cross-study prediction to assess model generalizability.

    :param dataset: Held-out dataset from another study.
    :param model: Trained model instance to evaluate.
    :param test_mode: Split mode used for overlap removal.
    :param train_dataset: Training dataset from the source study.
    :param path_data: Root directory for feature tables.
    :param early_stopping_dataset: Optional early-stopping data for retraining.
    :param response_transformation: Optional response transformer.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param single_drug_id: Drug identifier when *model* is single-drug scoped.
    """
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
    """Resolve feature views to randomize for stress tests.

    :param model_class: Model class whose featurizers define available views.
    :param randomization_mode: Requested randomization modes (for example ``SVCC``).
    :param hyperparameters: Model hyperparameters used to resolve view names.

    :returns: Mapping from test names to feature-view lists.
    """
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
    """Run randomization stress tests for one CV fold.

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
    """Randomize feature views, then train and predict once.

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
    """Run robustness tests for one CV fold.

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
    """Train and predict for one robustness-test trial.

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
    """Split a validation set into validation and early-stopping partitions.

    :param validation_dataset: Validation dataset to subdivide.
    :param test_mode: One of ``LPO``, ``LCO``, ``LDO``, or ``LTO``.

    :returns: Validation and early-stopping datasets.
    """
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
    """Train the model and predict the response for the prediction dataset.

    :param model: Trained or untrained ``DRPModel`` instance.
    :param path_data: Root directory for feature tables.
    :param train_dataset: Training responses and identifiers.
    :param prediction_dataset: Pairs to predict; receives predictions in place.
    :param early_stopping_dataset: Optional hold-out for early stopping.
    :param response_transformation: Optional sklearn response transformer.
    :param cl_features: Preloaded cell-line features, or ``None`` to load from disk.
    :param drug_features: Preloaded drug features, or ``None`` to load from disk.
    :param model_checkpoint_dir: Directory for predictor checkpoints.

    :returns: *prediction_dataset* with ``predictions`` populated.
    """
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
    """Train a model and compute validation metrics.

    :param model: Model instance to train.
    :param path_data: Root directory for feature tables.
    :param train_dataset: Training split.
    :param validation_dataset: Validation split to score.
    :param early_stopping_dataset: Optional early-stopping data.
    :param response_transformation: Optional response transformer.
    :param metric: Primary metric to optimize and return.
    :param model_checkpoint_dir: Directory for model checkpoints.

    :returns: Validation metrics keyed by metric name.
    """
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
    """Build experiment run keys for multi- and single-drug models.

    :param models: Model classes to include in the run.
    :param response_data: Dataset used to enumerate single-drug keys.

    :returns: Mapping from run key to base model name.
    """
    return _make_model_list(models, response_data)


@pipeline_function
def get_model_name_and_drug_id(model_name: str) -> tuple[str, str | None]:
    """Parse a run key into model name and optional drug id.

    :param model_name: Run key, optionally suffixed with ``.<drug_id>``.

    :returns: Base model name and drug id, or ``None`` for multi-drug models.
    """
    return _get_model_name_and_drug_id(model_name)


@pipeline_function
def generate_data_saving_path(model_name, drug_id, result_path, suffix) -> str:
    """Return an output directory for predictions, HPO, and similar artifacts.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :param suffix: Subdirectory label (for example ``predictions``).

    :returns: Created output directory path.
    """
    return _generate_data_saving_path(model_name, drug_id, result_path, suffix)


@pipeline_function
def generate_final_model_checkpoint_path(model_name, drug_id, result_path) -> str:
    """Return archive path stem for a final production model checkpoint.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :returns: Checkpoint path stem; ``save_model`` appends ``.zip`` when missing.
    """
    return _generate_final_model_checkpoint_path(model_name, drug_id, result_path)


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
    """Train and persist a final production model on the full dataset.

    :param model_class: Model class to train.
    :param full_dataset: Complete response dataset for final training.
    :param response_transformation: Response transformer fitted on training data.
    :param path_data: Root directory for feature tables.
    :param model_checkpoint_dir: Directory for intermediate checkpoints.
    :param metric: Metric optimized during optional hyperparameter tuning.
    :param final_model_path: Archive path stem for the final model (``.zip`` appended on save).
    :param test_mode: Split mode for the internal train/validation holdout.
    :param val_ratio: Validation fraction for the holdout split.
    :param hyperparameter_tuning: Whether to tune hyperparameters before training.
    :param hpo_num_samples: Number of HPO trials when tuning is enabled.
    :param hpo_random_state: Random seed for hyperparameter search.
    :param hpo_resources_per_trial: Ray resource allocation per HPO trial.
    :param hpo_storage_path: Optional Ray Tune storage path for HPO results.
    """
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
    """Split a dataset into train and validation sets.

    :param dataset: Full dataset to split.
    :param test_mode: One of ``LPO``, ``LCO``, ``LDO``, or ``LTO``.
    :param val_ratio: Approximate validation fraction.
    :param random_state: Random seed for splitting.

    :returns: Train and validation datasets.
    """
    return make_train_val_split_impl(dataset, test_mode, val_ratio, random_state)


# Ray Tune entry points and tests import ``hpam_tune`` from ``drevalpy.experiment``.
hpam_tune = hpam_tune
