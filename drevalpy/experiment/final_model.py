"""Final production model training for experiment workflows."""

from __future__ import annotations

from pathlib import Path

from sklearn.base import TransformerMixin, clone

from ..datasets.dataset import DrugResponseDataset, split_early_stopping_data
from ..models.drp_model import DRPModel
from ..utils.checkpoints import checkpoint_dir_or_temporary
from .fold import make_train_val_split_impl, merge_train_validation
from .hpo import select_final_model_hyperparameters


def _prepare_final_train_val(
    full_dataset: DrugResponseDataset,
    test_mode: str,
    val_ratio: float,
    model_class: type[DRPModel],
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset | None]:
    full_dataset.remove_nan_responses()
    train_dataset, validation_dataset = make_train_val_split_impl(
        full_dataset, test_mode=test_mode, val_ratio=val_ratio
    )
    if model_class.supports_early_stopping():
        validation_dataset, early_stopping_dataset = split_early_stopping_data(validation_dataset, test_mode)
    else:
        early_stopping_dataset = None
    return train_dataset, validation_dataset, early_stopping_dataset


def _reduce_final_training_corpus(
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    cell_lines_to_keep,
    drugs_to_keep,
    fold_transform: TransformerMixin | None,
) -> tuple[DrugResponseDataset, DrugResponseDataset | None]:
    train_dataset = merge_train_validation(train_dataset, validation_dataset)
    len_train_before = len(train_dataset)
    train_dataset = train_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if len(train_dataset) < len_train_before:
        print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}, due to missing features")

    if fold_transform is None:
        return train_dataset, early_stopping_dataset

    train_dataset = train_dataset.fit_transformed(fold_transform)
    if early_stopping_dataset is None:
        return train_dataset, None

    len_early_stopping_before = len(early_stopping_dataset)
    early_stopping_dataset = early_stopping_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if len(early_stopping_dataset) < len_early_stopping_before:
        print(
            f"Reduced early stopping dataset from {len_early_stopping_before} to "
            f"{len(early_stopping_dataset)}, due to missing features"
        )
    early_stopping_dataset = early_stopping_dataset.transformed(fold_transform)
    return train_dataset, early_stopping_dataset


def train_final_model_impl(
    model_class: type[DRPModel],
    full_dataset: DrugResponseDataset,
    response_transformation: TransformerMixin,
    model_checkpoint_dir: str | Path | None,
    metric: str,
    final_model_path: str | Path,
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
    :param model_checkpoint_dir: Directory for intermediate checkpoints, or ``None`` for a temporary one.
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
    from drevalpy.components.tuning.config import build_experiment_hpo_config

    print("Training final model with application-specific validation strategy ...")
    train_dataset, validation_dataset, early_stopping_dataset = _prepare_final_train_val(
        full_dataset, test_mode, val_ratio, model_class
    )

    hpo_cfg = build_experiment_hpo_config(
        metric,
        n_trials=hpo_num_samples,
        random_state=hpo_random_state,
        resources_per_trial=hpo_resources_per_trial,
        storage_path=hpo_storage_path,
    )
    best_hpams = select_final_model_hyperparameters(
        model_class=model_class,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
        metric=metric,
        model_checkpoint_dir=model_checkpoint_dir,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_config=hpo_cfg,
    )

    print(f"Best hyperparameters for final model: {best_hpams}")
    model = model_class(best_hpams)

    cl_features = model.load_cell_line_features(dataset_name=full_dataset.dataset_name)
    drug_features = model.load_drug_features(dataset_name=full_dataset.dataset_name)
    cell_lines_to_keep = cl_features.identifiers
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None

    fold_transform = clone(response_transformation) if response_transformation is not None else None
    train_dataset, early_stopping_dataset = _reduce_final_training_corpus(
        train_dataset,
        validation_dataset,
        early_stopping_dataset,
        cell_lines_to_keep,
        drugs_to_keep,
        fold_transform,
    )

    drug_features_copy = drug_features.copy() if drug_features is not None else None
    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        model.train(
            output=train_dataset,
            output_earlystopping=early_stopping_dataset,
            cell_line_input=cl_features.copy(),
            drug_input=drug_features_copy,
            model_checkpoint_dir=checkpoint_dir,
        )
    if fold_transform is not None:
        train_dataset.inverse_transform(fold_transform)
        if early_stopping_dataset is not None:
            early_stopping_dataset.inverse_transform(fold_transform)

    final_model_target = Path(final_model_path)
    final_model_target.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_target)
