"""Training and prediction helpers for experiment workflows."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import TransformerMixin

from ..datasets.dataset import DrugResponseDataset, FeatureDataset
from ..models.drp_model import DRPModel

if TYPE_CHECKING:
    pass


def load_features(
    model: DRPModel, path_data: str, dataset: DrugResponseDataset
) -> tuple[FeatureDataset, FeatureDataset | None]:
    """Load and reduce cell line and drug features for a given dataset.

    :param model: Model used to load feature views.
    :param path_data: Root data directory (for example ``data/``).
    :param dataset: Dataset whose ``dataset_name`` selects feature tables.

    :returns: Cell-line features and optional drug features.
    """
    cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=dataset.dataset_name)
    drug_features = model.load_drug_features(data_path=path_data, dataset_name=dataset.dataset_name)
    return cl_features, drug_features


def _copy_fold_datasets(
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset | None]:
    train_copy = train_dataset.copy()
    pred_copy = prediction_dataset.copy()
    es_copy = early_stopping_dataset.copy() if early_stopping_dataset is not None else None
    return train_copy, pred_copy, es_copy


def _resolve_feature_matrices(
    model: DRPModel,
    path_data: str,
    train_dataset: DrugResponseDataset,
    cl_features: FeatureDataset | None,
    drug_features: FeatureDataset | None,
) -> tuple[FeatureDataset | None, FeatureDataset | None]:
    if cl_features is None:
        print("Loading cell line features ...")
        cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    if drug_features is None:
        print("Loading drug features ...")
        drug_features = model.load_drug_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    return cl_features, drug_features


def _log_feature_coverage(
    train_dataset: DrugResponseDataset,
    cell_lines_to_keep: np.ndarray | None,
    drugs_to_keep: np.ndarray | None,
) -> None:
    if cell_lines_to_keep is not None:
        print(f"Number of cell lines in features: {len(cell_lines_to_keep)}")
    if drugs_to_keep is not None:
        print(f"Number of drugs in features: {len(drugs_to_keep)}")
    print(f"Number of cell lines in train dataset: {len(np.unique(train_dataset.cell_line_ids))}")
    print(f"Number of drugs in train dataset: {len(np.unique(train_dataset.drug_ids))}")


def _reduce_datasets_to_features(
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    cell_lines_to_keep: np.ndarray | None,
    drugs_to_keep: np.ndarray | None,
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset | None]:
    len_train_before = len(train_dataset)
    len_pred_before = len(prediction_dataset)
    train_reduced = train_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    pred_reduced = prediction_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if len(train_reduced) < len_train_before or len(pred_reduced) < len_pred_before:
        print(f"Reduced training dataset from {len_train_before} to {len(train_reduced)}, due to missing features")
        print(f"Reduced prediction dataset from {len_pred_before} to {len(pred_reduced)}, due to missing features")

    if early_stopping_dataset is None:
        return train_reduced, pred_reduced, None

    len_es_before = len(early_stopping_dataset)
    es_reduced = early_stopping_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced early stopping dataset from {len_es_before} to {len(es_reduced)}")
    return train_reduced, pred_reduced, es_reduced


def _apply_response_transform(
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    fold_transform: TransformerMixin | None,
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset | None]:
    if fold_transform is None:
        return train_dataset, prediction_dataset, early_stopping_dataset
    train_t = train_dataset.fit_transformed(fold_transform)
    pred_t = prediction_dataset.transformed(fold_transform)
    es_t = early_stopping_dataset.transformed(fold_transform) if early_stopping_dataset is not None else None
    return train_t, pred_t, es_t


def _train_model_with_checkpoints(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    cl_features: FeatureDataset,
    drug_features: FeatureDataset | None,
    model_checkpoint_dir: str,
) -> None:
    drug_input = drug_features.copy() if drug_features is not None else None
    print("Training model ...")
    if model_checkpoint_dir == "TEMPORARY":
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir} for model checkpoints")
            model.train(
                output=train_dataset,
                output_earlystopping=early_stopping_dataset,
                cell_line_input=cl_features.copy(),
                drug_input=drug_input,
                model_checkpoint_dir=temp_dir,
            )
        return
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir, exist_ok=True)
    print(f"Using directory: {model_checkpoint_dir} for model checkpoints")
    model.train(
        output=train_dataset,
        output_earlystopping=early_stopping_dataset,
        cell_line_input=cl_features.copy(),
        drug_input=drug_input,
        model_checkpoint_dir=model_checkpoint_dir,
    )


def _run_predictions(
    model: DRPModel,
    prediction_dataset: DrugResponseDataset,
    cl_features: FeatureDataset,
    drug_features: FeatureDataset | None,
) -> DrugResponseDataset:
    if len(prediction_dataset) == 0:
        prediction_dataset._predictions = np.array([])
        return prediction_dataset
    drug_input = drug_features.copy() if drug_features is not None else None
    prediction_dataset._predictions = model.predict(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features.copy(),
        drug_input=drug_input,
    )
    return prediction_dataset


def _inverse_transform_fold(
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    fold_transform: TransformerMixin | None,
) -> None:
    if fold_transform is None:
        return
    train_dataset.inverse_transform(fold_transform)
    prediction_dataset.inverse_transform(fold_transform)
    if early_stopping_dataset is not None:
        early_stopping_dataset.inverse_transform(fold_transform)


def train_and_predict_impl(
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

    :raises ValueError: If ``train_dataset`` has no ``dataset_name`` or cell-line features are missing.
    """
    train_dataset, prediction_dataset, early_stopping_dataset = _copy_fold_datasets(
        train_dataset, prediction_dataset, early_stopping_dataset
    )
    fold_transform = response_transformation

    if train_dataset.dataset_name is None:
        raise ValueError("train_dataset must have a dataset_name")

    cl_features, drug_features = _resolve_feature_matrices(model, path_data, train_dataset, cl_features, drug_features)
    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None
    _log_feature_coverage(train_dataset, cell_lines_to_keep, drugs_to_keep)

    train_dataset, prediction_dataset, early_stopping_dataset = _reduce_datasets_to_features(
        train_dataset,
        prediction_dataset,
        early_stopping_dataset,
        cell_lines_to_keep,
        drugs_to_keep,
    )
    train_dataset, prediction_dataset, early_stopping_dataset = _apply_response_transform(
        train_dataset, prediction_dataset, early_stopping_dataset, fold_transform
    )

    if cl_features is None:
        raise ValueError("cell line features are required for training")
    _train_model_with_checkpoints(
        model,
        train_dataset,
        early_stopping_dataset,
        cl_features,
        drug_features,
        model_checkpoint_dir,
    )
    prediction_dataset = _run_predictions(model, prediction_dataset, cl_features, drug_features)
    _inverse_transform_fold(train_dataset, prediction_dataset, early_stopping_dataset, fold_transform)
    return prediction_dataset
