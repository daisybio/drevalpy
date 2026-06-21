"""Bridge existing :class:`~drevalpy.models.drp_model.DRPModel` classes to component configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.components.composed_model import ComposedModel
from drevalpy.components.config import ModelConfig, PredictionMode
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

if TYPE_CHECKING:
    from drevalpy.models.baselines.naive_pred import NaiveModel
    from drevalpy.models.baselines.sklearn_models import SklearnModel


class ComponentDRPBridge:
    """Shared train/predict logic for DRP models backed by :class:`ComposedModel`."""

    def __init__(self) -> None:
        self._composed: ComposedModel | None = None
        self._needs_tissue: bool = False

    def set_composed_config(self, config: ModelConfig, *, needs_tissue: bool = False) -> None:
        self._needs_tissue = needs_tissue
        self._composed = config.create_model()

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        if self._composed is None:
            msg = "Component config has not been built"
            raise RuntimeError(msg)
        tissue_input = cell_line_input if self._needs_tissue else None
        if getattr(self._composed._predictor, "uses_raw_features", False):
            self._composed._predictor.fit_raw(
                output,
                cell_line_input,
                drug_input,
                output_earlystopping=output_earlystopping,
            )
        else:
            self._composed.train(
                output,
                cell_line_input,
                drug_input,
                tissue_input=tissue_input,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._composed is None:
            return np.full(len(cell_line_ids), np.nan)
        return self._composed.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    @property
    def composed(self) -> ComposedModel | None:
        return self._composed

    def is_trained(self) -> bool:
        if self._composed is None:
            return False
        predictor = self._composed._predictor
        if getattr(predictor, "uses_raw_features", False):
            return getattr(predictor, "_model", None) is not None
        if not predictor.uses_features:
            return getattr(predictor, "_dataset_mean", None) is not None
        return getattr(predictor, "_estimator", None) is not None


def ensure_components_registered() -> None:
    """Import built-in featurizers and predictors so registry decorators run."""
    from drevalpy.components.register_builtins import ensure_components_registered as _ensure

    _ensure()


def preview_sklearn_estimator(bridge: ComponentDRPBridge, hyperparameters: dict[str, Any]) -> Any:
    """Build an unfitted sklearn estimator for pre-train inspection."""
    composed = bridge.composed
    if composed is None:
        return None
    predictor = composed._predictor
    if not hasattr(predictor, "_make_estimator"):
        return None
    merged_hp = {
        **predictor.get_default_hyperparameters(),
        **hyperparameters,
        "prediction_mode": PredictionMode.REGRESSION,
    }
    predictor.build(merged_hp, {"cell_line": 1, "drug": 1, "n_classes": 1})
    return predictor._make_estimator()


def sync_sklearn_from_components(model: SklearnModel) -> None:
    """Copy fitted sklearn and preprocessing state from the composed model."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    if hasattr(predictor, "_estimator"):
        model.model = predictor._estimator

    cell_line_featurizer = composed._cell_line_featurizer
    if cell_line_featurizer is None:
        return

    from drevalpy.components.featurizers.cell_line.multi_concat import MultiConcatCellLineFeaturizer
    from drevalpy.components.featurizers.cell_line.view import ProteomicsCellLineFeaturizer, ScaledGeneExpressionFeaturizer

    if isinstance(cell_line_featurizer, ScaledGeneExpressionFeaturizer):
        model.gene_expression_scaler = cell_line_featurizer._scaler
    elif isinstance(cell_line_featurizer, ProteomicsCellLineFeaturizer):
        model.proteomics_transformer = cell_line_featurizer._transformer
    elif isinstance(cell_line_featurizer, MultiConcatCellLineFeaturizer):
        model.gene_expression_scaler = cell_line_featurizer._gene_expression_scaler
        model.methylation_scaler = cell_line_featurizer._methylation_scaler
        model.methylation_pca = cell_line_featurizer._methylation_pca
        if "proteomics" in cell_line_featurizer._views:
            model.proteomics_transformer = cell_line_featurizer._proteomics_transformer


def restore_sklearn_to_components(model: SklearnModel) -> None:
    """Inject serialized sklearn and preprocessing state into the composed model."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    if hasattr(predictor, "_estimator"):
        predictor._estimator = model.model

    cell_line_featurizer = composed._cell_line_featurizer
    if cell_line_featurizer is None:
        return

    from drevalpy.components.featurizers.cell_line.multi_concat import MultiConcatCellLineFeaturizer
    from drevalpy.components.featurizers.cell_line.view import ProteomicsCellLineFeaturizer, ScaledGeneExpressionFeaturizer

    if isinstance(cell_line_featurizer, ScaledGeneExpressionFeaturizer) and model.gene_expression_scaler is not None:
        cell_line_featurizer._scaler = model.gene_expression_scaler
        cell_line_featurizer._fitted_features = object()
    elif isinstance(cell_line_featurizer, ProteomicsCellLineFeaturizer) and model.proteomics_transformer is not None:
        cell_line_featurizer._transformer = model.proteomics_transformer
    elif isinstance(cell_line_featurizer, MultiConcatCellLineFeaturizer):
        cell_line_featurizer._gene_expression_scaler = model.gene_expression_scaler
        if model.methylation_scaler is not None:
            cell_line_featurizer._methylation_scaler = model.methylation_scaler
        if model.methylation_pca is not None:
            cell_line_featurizer._methylation_pca = model.methylation_pca
        if model.proteomics_transformer is not None:
            cell_line_featurizer._proteomics_transformer = model.proteomics_transformer
        cell_line_featurizer._train_ids = np.array([], dtype=str)


_NAIVE_STATE_ATTRS = {
    "naiveMean": [("_dataset_mean", "dataset_mean")],
    "naiveDrugMean": [
        ("_dataset_mean", "dataset_mean"),
        ("_entity_means", "drug_means"),
    ],
    "naiveCellLineMean": [
        ("_dataset_mean", "dataset_mean"),
        ("_entity_means", "cell_line_means"),
    ],
    "naiveTissueMean": [
        ("_dataset_mean", "dataset_mean"),
        ("_entity_means", "tissue_means"),
    ],
    "naiveTissueDrugMean": [
        ("_dataset_mean", "dataset_mean"),
        ("_combo_means", "tissue_drug_means"),
    ],
    "naiveMeanEffects": [
        ("_dataset_mean", "dataset_mean"),
        ("_cell_line_effects", "cell_line_effects"),
        ("_drug_effects", "drug_effects"),
    ],
}


def sync_naive_from_components(model: NaiveModel, predictor_type: str) -> None:
    """Copy naive predictor state to legacy DRPModel attributes."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    for pred_attr, model_attr in _NAIVE_STATE_ATTRS.get(predictor_type, []):
        if hasattr(predictor, pred_attr):
            value = getattr(predictor, pred_attr)
            if pred_attr == "_combo_means":
                value = {tuple(key.split("|", maxsplit=1)): mean for key, mean in value.items()}
            setattr(model, model_attr, value)


def restore_naive_to_components(model: NaiveModel, predictor_type: str) -> None:
    """Inject legacy naive state into the composed predictor."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    for pred_attr, model_attr in _NAIVE_STATE_ATTRS.get(predictor_type, []):
        if hasattr(model, model_attr):
            value = getattr(model, model_attr)
            if pred_attr == "_combo_means":
                value = {
                    f"{tissue}|{drug}": mean
                    for (tissue, drug), mean in value.items()
                }
            setattr(predictor, pred_attr, value)
