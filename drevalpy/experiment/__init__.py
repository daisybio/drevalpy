"""Cross-validation loops and experiment orchestration entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.base import TransformerMixin

from ..components.tuning.hpo import hpam_tune  # noqa: F401
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


def _cross_study_prediction_legacy(
    dataset,
    model: DRPModel,
    test_mode: str,
    train_dataset,
    early_stopping_dataset,
    response_transformation,
    path_out: str | Path,
    split_index: int,
    single_drug_id: str | None,
) -> None:
    """Legacy cross-study prediction using DrugResponseDataset objects."""
    import warnings

    import numpy as np

    dataset = dataset.copy()
    out_dir = Path(path_out) / "cross_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    if response_transformation:
        dataset.transform(response_transformation)

    try:
        from drevalpy.components.feature_source import CellLineFeatureSource, DrugFeatureSource
        from drevalpy.datasets import load_mudataset

        mudataset = load_mudataset(dataset.dataset_name)
        all_cl_ids = np.array(mudataset.cell_line_ids)
        all_drug_ids = np.array(mudataset.drug_ids)
        cl_features = CellLineFeatureSource(mudataset, all_cl_ids)
        drug_features = DrugFeatureSource(mudataset, all_drug_ids)
    except (ValueError, FileNotFoundError) as e:
        warnings.warn(str(e), stacklevel=2)
        return

    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None
    if single_drug_id is not None:
        drugs_to_keep = np.array([single_drug_id])

    dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    if early_stopping_dataset is not None:
        train_dataset = train_dataset.with_rows_added(early_stopping_dataset)

    _remove_train_overlap(test_mode, train_dataset, dataset)

    if len(dataset) == 0:
        warnings.warn("No samples remaining after overlap removal for cross-study dataset.", stacklevel=2)
        return

    if not model._stack.is_fitted():
        warnings.warn("Model was not trained (empty training set); skipping cross-study prediction.", stacklevel=2)
        return

    dataset.shuffle(random_state=42)
    drug_input = drug_features.copy() if drug_features is not None else None
    dataset._predictions = model.predict(
        cell_line_ids=dataset.cell_line_ids,
        drug_ids=dataset.drug_ids,
        cell_line_input=cl_features.copy(),
        drug_input=drug_input,
    )
    if response_transformation:
        dataset.inverse_transform(response_transformation)

    dataset.to_csv(out_dir / f"cross_study_{dataset.dataset_name}_split_{split_index}.csv")


def _remove_train_overlap(
    test_mode: str,
    train_dataset,
    dataset,
) -> None:
    """Remove overlap between train and cross-study dataset in place."""
    import numpy as np

    handlers = {
        "LPO": _remove_overlap_lpo,
        "LCO": _remove_overlap_lco,
        "LDO": _remove_overlap_ldo,
        "LTO": _remove_overlap_lto,
    }
    handler = handlers.get(test_mode)
    if handler is None:
        raise ValueError(f"Invalid test mode: {test_mode}. Choose from LPO, LCO, LDO, LTO")
    handler(train_dataset, dataset, np)


def _remove_overlap_lpo(train_dataset, dataset, np) -> None:
    train_pairs = {f"{cl}_{drug}" for cl, drug in zip(train_dataset.cell_line_ids, train_dataset.drug_ids, strict=True)}
    dataset_pairs = [f"{cl}_{drug}" for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids, strict=True)]
    dataset.remove_rows(np.array([i for i, pair in enumerate(dataset_pairs) if pair in train_pairs]))


def _remove_overlap_lco(train_dataset, dataset, np) -> None:
    dataset.reduce_to(
        cell_line_ids=np.setdiff1d(dataset.cell_line_ids, train_dataset.cell_line_ids),
        drug_ids=None,
    )


def _remove_overlap_ldo(train_dataset, dataset, np) -> None:
    dataset.reduce_to(
        cell_line_ids=None,
        drug_ids=np.setdiff1d(dataset.drug_ids, train_dataset.drug_ids),
    )


def _remove_overlap_lto(train_dataset, dataset, np) -> None:
    if train_dataset.tissue is None or dataset.tissue is None:
        raise ValueError("Tissue information not available.")
    train_tissues = set(train_dataset.tissue)
    indices = np.array([i for i, t in enumerate(dataset.tissue) if t not in train_tissues])
    cell_lines_to_keep = np.unique(dataset.cell_line_ids[indices]) if len(indices) > 0 else np.array([])
    dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=None)


@pipeline_function
def cross_study_prediction(
    target=None,
    model: DRPModel | None = None,
    test_mode: str = "LPO",
    train_masks: SplitMasks | None = None,
    source: MuDataset | None = None,
    path_out: str | Path = ".",
    split_index: int = 0,
    dataset_name: str = "cross_study",
    *,
    dataset=None,
    train_dataset=None,
    early_stopping_dataset=None,
    response_transformation=None,
    single_drug_id: str | None = None,
) -> None:
    """Run cross-study prediction to assess model generalizability.

    Supports both the new MuDataset signature and the legacy DrugResponseDataset
    signature for backward compatibility.
    """
    # Legacy DrugResponseDataset path
    if dataset is not None or (target is not None and not isinstance(target, MuDataset)):
        _cross_study_prediction_legacy(
            dataset=dataset if dataset is not None else target,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            early_stopping_dataset=early_stopping_dataset,
            response_transformation=response_transformation,
            path_out=path_out,
            split_index=split_index,
            single_drug_id=single_drug_id,
        )
        return

    # New MuDataset path
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


def _train_and_predict_on_features(
    model: DRPModel,
    train_dataset,
    validation_dataset,
    early_stopping_dataset,
    cl_features,
    drug_features,
    model_checkpoint_dir: str | Path | None,
) -> None:
    """Train model and populate validation_dataset._predictions in place."""
    import numpy as np

    from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary

    drug_input = drug_features.copy() if drug_features is not None else None
    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        model.train(
            output=train_dataset,
            cell_line_input=cl_features.copy(),
            drug_input=drug_input,
            output_earlystopping=early_stopping_dataset,
            model_checkpoint_dir=checkpoint_dir,
        )

    if len(validation_dataset) == 0:
        validation_dataset._predictions = np.array([])
    elif not model._stack.is_fitted():
        validation_dataset._predictions = np.full(len(validation_dataset), np.nan)
    else:
        drug_input = drug_features.copy() if drug_features is not None else None
        validation_dataset._predictions = model.predict(
            cell_line_ids=validation_dataset.cell_line_ids,
            drug_ids=validation_dataset.drug_ids,
            cell_line_input=cl_features.copy(),
            drug_input=drug_input,
        )


def train_and_evaluate(
    model: DRPModel,
    train_dataset: Any,
    validation_dataset: Any,
    early_stopping_dataset: Any | None = None,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    model_checkpoint_dir: str | Path | None = None,
) -> dict[str, float]:
    """Train a model and compute validation metrics (legacy compat for HPO runtime).

    :param model: Model instance to train.
    :param train_dataset: Training split (DrugResponseDataset).
    :param validation_dataset: Validation split to score.
    :param early_stopping_dataset: Optional early-stopping data.
    :param response_transformation: Optional response transformer.
    :param metric: Primary metric to optimize and return.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :returns: Validation metrics keyed by metric name.
    """
    from sklearn.base import clone

    trial_transform = None if response_transformation is None else clone(response_transformation)

    train_dataset = train_dataset.copy()
    validation_dataset = validation_dataset.copy()
    early_stopping_dataset = early_stopping_dataset.copy() if early_stopping_dataset is not None else None

    if train_dataset.dataset_name is None:
        raise ValueError("train_dataset must have a dataset_name")

    import numpy as np

    from drevalpy.components.feature_source import CellLineFeatureSource, DrugFeatureSource
    from drevalpy.datasets import load_mudataset

    mudataset = load_mudataset(train_dataset.dataset_name)
    all_cl_ids = np.array(mudataset.cell_line_ids)
    all_drug_ids = np.array(mudataset.drug_ids)
    cl_features = CellLineFeatureSource(mudataset, all_cl_ids)
    drug_features = DrugFeatureSource(mudataset, all_drug_ids)

    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None

    train_dataset = train_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    validation_dataset = validation_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if early_stopping_dataset is not None:
        early_stopping_dataset = early_stopping_dataset.reduced_to(
            cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep
        )

    if trial_transform is not None:
        train_dataset = train_dataset.fit_transformed(trial_transform)
        validation_dataset = validation_dataset.transformed(trial_transform)
        if early_stopping_dataset is not None:
            early_stopping_dataset = early_stopping_dataset.transformed(trial_transform)

    _train_and_predict_on_features(
        model,
        train_dataset,
        validation_dataset,
        early_stopping_dataset,
        cl_features,
        drug_features,
        model_checkpoint_dir,
    )

    if trial_transform is not None:
        train_dataset.inverse_transform(trial_transform)
        validation_dataset.inverse_transform(trial_transform)
        if early_stopping_dataset is not None:
            early_stopping_dataset.inverse_transform(trial_transform)

    additional_metrics = None
    if metric not in ["R^2", "Pearson"]:
        additional_metrics = [metric]
    return model.compute_and_log_final_metrics(
        predictions=validation_dataset.predictions,
        response=validation_dataset.response,
        additional_metrics=additional_metrics,
        prefix="val_",
    )
