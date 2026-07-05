"""Bridge `~drevalpy.models.drp_model.DRPModel` adapters to modular component configs."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import ModelConfig, PredictionMode

if TYPE_CHECKING:
    from drevalpy.components.predictors.baselines.naive_pred import NaiveModel
    from drevalpy.components.predictors.baselines.sklearn_models import SklearnModel
    from drevalpy.models.composed_model import ComposedModel

_SKLEARN_FEATURIZER_STATE_ATTRS = (
    "gene_expression_scaler",
    "proteomics_transformer",
    "methylation_scaler",
    "methylation_pca",
)

_COMPONENT_STACK_FILE = "component_stack.joblib"
_HYPERPARAMETERS_FILE = "hyperparameters.json"


def _featurizer_state(featurizer: Any) -> dict[str, object]:
    if featurizer is None:
        return {}
    from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer

    if isinstance(featurizer, ConcatFeaturizersCellLineFeaturizer):
        return ConcatFeaturizersCellLineFeaturizer.collect_legacy_state(featurizer)
    return featurizer.get_state()


def _restore_featurizer_state(featurizer: Any, state: dict[str, object]) -> None:
    if featurizer is None or not state:
        return
    from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer

    if isinstance(featurizer, ConcatFeaturizersCellLineFeaturizer):
        ConcatFeaturizersCellLineFeaturizer.distribute_legacy_state(featurizer, state)
    else:
        featurizer.set_state(state)


def save_component_stack(
    bridge: ComponentDRPBridge,
    directory: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> None:
    """Persist composed featurizer and predictor state."""
    composed = bridge.composed
    if composed is None or not bridge.is_trained():
        msg = "Cannot save: component stack is not trained"
        raise RuntimeError(msg)
    os.makedirs(directory, exist_ok=True)
    stack = {
        "predictor": composed._predictor.get_state(),
        "cell_line_featurizer": _featurizer_state(composed._cell_line_featurizer),
        "drug_featurizer": _featurizer_state(composed._drug_featurizer),
    }
    joblib.dump(stack, os.path.join(directory, _COMPONENT_STACK_FILE))
    if hyperparameters is not None:
        with open(os.path.join(directory, _HYPERPARAMETERS_FILE), "w") as handle:
            json.dump(hyperparameters, handle)


def load_component_stack(bridge: ComponentDRPBridge, directory: str) -> dict[str, Any]:
    """Restore composed featurizer and predictor state."""
    composed = bridge.composed
    if composed is None:
        msg = "Cannot load: component stack has not been built"
        raise RuntimeError(msg)
    stack_path = os.path.join(directory, _COMPONENT_STACK_FILE)
    if not os.path.exists(stack_path):
        msg = f"Missing component stack file: {stack_path}"
        raise FileNotFoundError(msg)
    stack = joblib.load(stack_path)
    composed._predictor.set_state(stack.get("predictor", {}))
    _restore_featurizer_state(composed._cell_line_featurizer, stack.get("cell_line_featurizer", {}))
    _restore_featurizer_state(composed._drug_featurizer, stack.get("drug_featurizer", {}))
    hyperparameters: dict[str, Any] = {}
    hyperparameters_path = os.path.join(directory, _HYPERPARAMETERS_FILE)
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path) as handle:
            hyperparameters = json.load(handle)
    return hyperparameters


class ComponentDRPBridge:
    """Shared train/predict logic for DRP models backed by `ComposedModel`."""

    def __init__(self) -> None:
        self._composed: ComposedModel | None = None

    def set_composed_config(self, config: ModelConfig) -> None:
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
                output_earlystopping=output_earlystopping,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if self._composed is None:
            msg = "Model has not been built; call build_model() before predict()"
            raise RuntimeError(msg)
        if not self.is_trained():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
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
        if hasattr(predictor, "is_fitted"):
            return predictor.is_fitted()
        return bool(predictor.get_state())


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


def _apply_sklearn_featurizer_state_to_model(model: SklearnModel, state: dict[str, object]) -> None:
    for attr in _SKLEARN_FEATURIZER_STATE_ATTRS:
        value = state.get(attr)
        if value is not None:
            setattr(model, attr, value)


def _sklearn_featurizer_state_from_model(model: SklearnModel) -> dict[str, object]:
    state: dict[str, object] = {}
    for attr in _SKLEARN_FEATURIZER_STATE_ATTRS:
        value = getattr(model, attr, None)
        if value is not None:
            state[attr] = value
    return state


def sync_sklearn_from_components(model: SklearnModel) -> None:
    """Copy fitted sklearn and preprocessing state from the composed model."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    predictor_state = predictor.get_state()
    estimator = predictor_state.get("estimator")
    if estimator is not None:
        model.model = estimator

    cell_line_featurizer = composed._cell_line_featurizer
    if cell_line_featurizer is not None:
        from drevalpy.components.featurizers.cell_line.concat import (
            ConcatFeaturizersCellLineFeaturizer,
        )

        if isinstance(cell_line_featurizer, ConcatFeaturizersCellLineFeaturizer):
            featurizer_state = ConcatFeaturizersCellLineFeaturizer.collect_legacy_state(
                cell_line_featurizer,
            )
        else:
            featurizer_state = cell_line_featurizer.get_state()
        _apply_sklearn_featurizer_state_to_model(model, featurizer_state)


def restore_sklearn_to_components(model: SklearnModel) -> None:
    """Inject serialized sklearn and preprocessing state into the composed model."""
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    if model.model is not None:
        predictor.set_state({**predictor.get_state(), "estimator": model.model})

    cell_line_featurizer = composed._cell_line_featurizer
    if cell_line_featurizer is not None:
        featurizer_state = _sklearn_featurizer_state_from_model(model)
        if featurizer_state:
            featurizer_state["fitted"] = True
            from drevalpy.components.featurizers.cell_line.concat import (
                ConcatFeaturizersCellLineFeaturizer,
            )

            if isinstance(cell_line_featurizer, ConcatFeaturizersCellLineFeaturizer):
                ConcatFeaturizersCellLineFeaturizer.distribute_legacy_state(
                    cell_line_featurizer,
                    featurizer_state,
                )
            else:
                cell_line_featurizer.set_state(featurizer_state)


def sync_literature_from_components(model: Any) -> None:
    """Copy fitted component state onto a literature DRPModel wrapper."""
    if hasattr(model, "_sync_impl_state_from_bridge"):
        model._sync_impl_state_from_bridge()


def restore_literature_to_components(model: Any) -> None:
    """Inject serialized literature model state into the composed stack."""
    from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
    from drevalpy.components.predictors.literature.public_models import LiteratureComponentDRPModel

    if not isinstance(model, LiteratureComponentDRPModel):
        return
    composed = model._bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    component_model = getattr(model, "model", None)
    if component_model is not None and hasattr(predictor, "_model"):
        predictor._model = component_model
    engine = getattr(predictor, "_engine", None)
    if engine is not None and component_model is None:
        for name, value in vars(model).items():
            if name.startswith("_") or name in {"hyperparameters", "wandb_project"}:
                continue
            if hasattr(engine, name):
                setattr(engine, name, value)

    cell_line_featurizer = composed._cell_line_featurizer
    if cell_line_featurizer is not None:
        featurizer_state: dict[str, object] = {"fitted": True}
        if hasattr(model, "gene_expression_scaler") and model.gene_expression_scaler is not None:
            featurizer_state["gene_expression_scaler"] = model.gene_expression_scaler
        if hasattr(model, "methylation_scaler") and model.methylation_scaler is not None:
            featurizer_state["methylation_scaler"] = model.methylation_scaler
        if hasattr(model, "methylation_pca") and model.methylation_pca is not None:
            featurizer_state["methylation_pca"] = model.methylation_pca
        if hasattr(model, "cell_line_views"):
            featurizer_state["views"] = list(model.cell_line_views)
        if hasattr(model, "input_dims") and isinstance(model.input_dims, dict):
            view_dims = {
                key: value for key, value in model.input_dims.items() if key not in getattr(model, "drug_views", [])
            }
            if view_dims:
                featurizer_state["view_dims"] = view_dims
        if isinstance(cell_line_featurizer, ConcatFeaturizersCellLineFeaturizer):
            ConcatFeaturizersCellLineFeaturizer.distribute_legacy_state(
                cell_line_featurizer,
                featurizer_state,
            )
        else:
            cell_line_featurizer.set_state(featurizer_state)


def sync_naive_from_components(model: NaiveModel, predictor_type: str) -> None:
    """Copy naive predictor state to legacy DRPModel attributes."""
    _ = predictor_type
    composed = model._component_bridge.composed
    if composed is None:
        return
    for key, value in composed._predictor.get_state().items():
        setattr(model, key, value)


def restore_naive_to_components(model: NaiveModel, predictor_type: str) -> None:
    """Inject legacy naive state into the composed predictor."""
    _ = predictor_type
    composed = model._component_bridge.composed
    if composed is None:
        return
    predictor = composed._predictor
    state: dict[str, object] = {}
    for key in (
        "dataset_mean",
        "drug_means",
        "cell_line_means",
        "tissue_means",
        "tissue_drug_means",
        "cell_line_effects",
        "drug_effects",
    ):
        if hasattr(model, key):
            value = getattr(model, key)
            if value is not None and value != {}:
                state[key] = value
    predictor.set_state(state)
