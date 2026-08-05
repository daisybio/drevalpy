"""Single-drug identity injection and scope inference for model configs."""

from __future__ import annotations

from typing import Any

from drevalpy.types.model_scope import ModelScope

from .featurizer import DrugFeaturizerConfig
from .predictor import PredictorConfig


def _infer_scope_for_predictor(pred_cls: type[Any]) -> ModelScope | None:
    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and len(supported_scopes) == 1:
        return next(iter(supported_scopes))
    return None


def _predictor_name_from_identity_data(data: dict[str, Any]) -> str | None:
    predictor = data.get("predictor")
    if predictor is None:
        return None
    if isinstance(predictor, PredictorConfig):
        return predictor.name
    if isinstance(predictor, dict):
        return str(predictor.get("name", ""))
    return str(predictor)


def _coerce_model_scope(scope: object) -> ModelScope:
    if isinstance(scope, ModelScope):
        return scope
    return ModelScope(str(scope))


def _maybe_apply_inferred_scope(data: dict[str, Any], pred_cls: type[Any]) -> tuple[dict[str, Any], ModelScope]:
    scope = _coerce_model_scope(data.get("scope", ModelScope.MULTI_DRUG))
    if "scope" in data:
        return data, scope
    inferred = _infer_scope_for_predictor(pred_cls)
    if inferred is None:
        return data, scope
    return {**data, "scope": inferred}, inferred


def _needs_identity_drug_featurizer(data: dict[str, Any], pred_cls: type[Any], scope: ModelScope) -> bool:
    if scope != ModelScope.SINGLE_DRUG:
        return False
    if data.get("cell_line_featurizer") is None:
        return False
    if data.get("drug_featurizer") is not None:
        return False
    return getattr(pred_cls, "routing_drug_featurizer", None) == "identity"


def normalize_single_drug_identity(data: dict[str, Any]) -> dict[str, Any]:
    """Inject implicit identity drug featurizer for single-drug feature-based configs.

    :param data: Raw model config mapping before pydantic validation.
    :returns: Mapping with optional ``drug_featurizer`` identity injection.
    """
    from drevalpy.components.registry import get_predictor

    predictor_name = _predictor_name_from_identity_data(data)
    if predictor_name is None:
        return data
    try:
        pred_cls = get_predictor(predictor_name)
    except (ValueError, ImportError):
        return data

    data, scope = _maybe_apply_inferred_scope(data, pred_cls)
    if not _needs_identity_drug_featurizer(data, pred_cls, scope):
        return data
    return {**data, "drug_featurizer": DrugFeaturizerConfig(name="identity")}
