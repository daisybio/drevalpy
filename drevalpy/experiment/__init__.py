"""Cross-validation loops and experiment orchestration entry points."""

from __future__ import annotations

from typing import Any

from sklearn.base import TransformerMixin
from upath import UPath as Path

from ..components.tuning.hpo import mu_hpam_tune  # noqa: F401
from ..datasets.mudataset import MuDataset
from ..datasets.splitting import EntityScope, MuDataSplitter, SplitMasks
from ..models.drp_model import DRPModel
from ..utils._pipeline_function import pipeline_function
from .cross_study import cross_study_prediction_impl
from .fold import MuFoldData, merge_train_val_scopes, prepare_mu_fold
from .model_paths import generate_data_saving_path as _generate_data_saving_path
from .model_paths import generate_final_model_checkpoint_path as _generate_final_model_checkpoint_path
from .model_paths import get_model_name_and_drug_id as _get_model_name_and_drug_id
from .run import mu_experiment
from .seed import seed_everything
from .splits import prepare_splits
from .training import mu_train_and_predict

_CWD = Path()

__all__ = [
    "EntityScope",
    "MuDataSplitter",
    "MuDataset",
    "MuFoldData",
    "SplitMasks",
    "consolidate_single_drug_model_predictions",
    "cross_study_prediction",
    "generate_data_saving_path",
    "generate_final_model_checkpoint_path",
    "get_model_name_and_drug_id",
    "get_randomization_test_views",
    "merge_train_val_scopes",
    "mu_experiment",
    "mu_train_and_predict",
    "prepare_mu_fold",
    "prepare_splits",
    "randomize_train_predict",
    "robustness_train_predict",
    "seed_everything",
]


@pipeline_function
def cross_study_prediction(
    target: MuDataset | None = None,
    model: DRPModel | None = None,
    test_mode: str = "LPO",
    train_masks: SplitMasks | None = None,
    source: MuDataset | None = None,
    path_out: str | Path = ".",
    split_index: int = 0,
    dataset_name: str = "cross_study",
) -> None:
    """Run cross-study prediction to assess model generalizability.

    :param target: Target MuDataset to predict on.
    :param model: Trained model.
    :param test_mode: Test mode (LPO, LCO, LDO, LTO).
    :param train_masks: Training split masks for overlap removal.
    :param source: Source MuDataset the model was trained on.
    :param path_out: Output directory for prediction files.
    :param split_index: CV fold index.
    :param dataset_name: Name to assign to the cross-study dataset.
    """
    cross_study_prediction_impl(
        target=target,
        model=model,
        test_mode=test_mode,
        train_masks=train_masks,
        source=source,
        path_out=path_out,
        split_index=split_index,
        dataset_name=dataset_name,
    )


@pipeline_function
def consolidate_single_drug_model_predictions(
    models: list[type[DRPModel]],
    n_cv_splits: int,
    results_path: str | Path,
    cross_study_datasets: list[str],
    randomization_mode: list[str] | None = None,
    n_trials_robustness: int = 0,
    out_path: str | Path = _CWD,
) -> None:
    """Consolidate per-fold single-drug predictions into summary files.

    :param models: Model classes whose outputs should be consolidated.
    :param n_cv_splits: Number of CV folds written during the experiment.
    :param results_path: Experiment result directory to read from.
    :param cross_study_datasets: Names of cross-study datasets to include.
    :param randomization_mode: Randomization views to consolidate, if any.
    :param n_trials_robustness: Number of robustness trials to consolidate.
    :param out_path: Output directory; defaults to the current working directory.
    """
    from .consolidate import consolidate_single_drug_model_predictions_impl

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
def get_model_name_and_drug_id(model_name: str) -> tuple[str, str | None]:
    """Parse a run key into model name and optional drug id.

    :param model_name: Run key, optionally suffixed with ``.<drug_id>``.
    :returns: Base model name and drug id, or ``None`` for multi-drug models.
    """
    return _get_model_name_and_drug_id(model_name)


@pipeline_function
def generate_data_saving_path(
    model_name: str,
    drug_id: str | None,
    result_path: str | Path,
    suffix: str,
) -> Path:
    """Return an output directory for predictions, HPO, and similar artifacts.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :param suffix: Subdirectory label (for example ``predictions``).
    :returns: Created output directory path.
    """
    return _generate_data_saving_path(model_name, drug_id, result_path, suffix)


@pipeline_function
def generate_final_model_checkpoint_path(
    model_name: str,
    drug_id: str | None,
    result_path: str | Path,
) -> Path:
    """Return archive path stem for a final production model checkpoint.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :returns: Checkpoint path stem.
    """
    return _generate_final_model_checkpoint_path(model_name, drug_id, result_path)


@pipeline_function
def randomize_train_predict(
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
    """Randomize feature views, then train and predict once.

    :param views: Feature view or views to randomize.
    :param test_name: Label for the randomization test output.
    :param randomization_type: Randomization strategy.
    :param randomization_test_file: Output path for predictions.
    :param model_class: Model class to train under randomized inputs.
    :param hyperparameters: Hyperparameters for model construction.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param response_transformation: Optional response transformer.
    """
    from .randomization import randomize_train_predict_impl

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
        model_checkpoint_dir=model_checkpoint_dir,
        response_transformation=response_transformation,
    )


@pipeline_function
def robustness_train_predict(
    trial: int,
    trial_file: str | Path,
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Train and predict for one robustness-test trial.

    :param trial: Trial index within the robustness test.
    :param trial_file: Output path for predictions.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param model_class: Model class to train on perturbed data.
    :param hyperparameters: Hyperparameters for model construction.
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints.
    """
    from .robustness import robustness_train_predict_impl

    robustness_train_predict_impl(
        trial=trial,
        trial_file=trial_file,
        mudataset=mudataset,
        train_scope=train_scope,
        test_scope=test_scope,
        early_stopping_scope=early_stopping_scope,
        model_class=model_class,
        hyperparameters=hyperparameters,
        response_transformation=response_transformation,
        model_checkpoint_dir=model_checkpoint_dir,
    )


@pipeline_function
def get_randomization_test_views(
    model_class: type[DRPModel],
    randomization_mode: list[str],
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Resolve feature views to randomize for stress tests.

    :param model_class: Model class whose featurizers define available views.
    :param randomization_mode: Requested randomization modes.
    :param hyperparameters: Model hyperparameters used to resolve view names.
    :returns: Mapping from test names to feature-view lists.
    """
    from .randomization import build_randomization_test_views

    return build_randomization_test_views(model_class, randomization_mode, hyperparameters)
