"""Read legacy DRPModel attributes from the component stack without mirroring state."""

from __future__ import annotations

from typing import Any

from drevalpy.components.state_helpers import state_str_list
from drevalpy.models._component_bridge import ComponentDRPBridge


def _composed_predictor(bridge: ComponentDRPBridge) -> Any | None:
    composed = bridge.composed
    if composed is None:
        return None
    return composed._predictor


def naive_state_from_bridge(bridge: ComponentDRPBridge) -> dict[str, Any]:
    """Return naive predictor state keys exposed on legacy wrappers."""
    predictor = _composed_predictor(bridge)
    if predictor is None:
        return {}
    return dict(predictor.get_state())


def sklearn_estimator_from_bridge(bridge: ComponentDRPBridge) -> Any | None:
    """Return the fitted sklearn estimator exposed by a legacy wrapper."""
    predictor = _composed_predictor(bridge)
    if predictor is None:
        return None
    state = predictor.get_state()
    return state.get("estimator")


def sklearn_featurizer_state_from_bridge(bridge: ComponentDRPBridge) -> dict[str, object]:
    """Return preprocessing state exposed by a legacy sklearn wrapper."""
    composed = bridge.composed
    if composed is None or composed._cell_line_featurizer is None:
        return {}
    from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
    from drevalpy.models._legacy_featurizer_state import collect_legacy_concat_state

    featurizer = composed._cell_line_featurizer
    if isinstance(featurizer, ConcatFeaturizersCellLineFeaturizer):
        return collect_legacy_concat_state(featurizer)
    return featurizer.get_state()


def literature_engine_from_bridge(bridge: ComponentDRPBridge) -> Any | None:
    """Return the underlying literature engine or model from the component stack."""
    predictor = _composed_predictor(bridge)
    if predictor is None:
        return None
    return getattr(predictor, "_engine", None) or getattr(predictor, "_model", None)


def literature_model_from_bridge(bridge: ComponentDRPBridge) -> Any | None:
    """Return the fitted literature model exposed by a legacy wrapper."""
    predictor = _composed_predictor(bridge)
    if predictor is None:
        return None
    return getattr(predictor, "_model", None)


def literature_featurizer_views_from_bridge(bridge: ComponentDRPBridge) -> list[str] | None:
    """Return legacy cell-line view names stored in featurizer state."""
    state = sklearn_featurizer_state_from_bridge(bridge)
    return state_str_list(state, "views")


def literature_input_dims_from_bridge(bridge: ComponentDRPBridge) -> dict[str, int] | None:
    """Return legacy per-view input dimensions derived from featurizer state."""
    state = sklearn_featurizer_state_from_bridge(bridge)
    view_dims = state.get("view_dims")
    if isinstance(view_dims, dict) and view_dims:
        return {str(key): int(value) for key, value in view_dims.items()}
    composed = bridge.composed
    if composed is None:
        return None
    cell_featurizer = composed._cell_line_featurizer
    if cell_featurizer is not None and hasattr(cell_featurizer, "_view"):
        return {cell_featurizer._view: int(cell_featurizer.output_dim)}
    return None
