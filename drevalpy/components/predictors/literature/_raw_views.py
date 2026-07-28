"""Validate raw FeatureDataset views for RawDatasetPredictor subclasses."""

from __future__ import annotations

from typing import Any

from drevalpy.datasets.dataset import FeatureDataset


def validate_required_views(
    features: FeatureDataset | None,
    required_views: tuple[str, ...],
    *,
    predictor_name: str,
    side: str,
) -> None:
    """Raise ``ValueError`` when required views are missing for any entity."""
    if not required_views:
        return
    if features is None:
        msg = f"{predictor_name} requires {side} FeatureDataset with views {list(required_views)}"
        raise ValueError(msg)
    if not features.features:
        msg = f"{predictor_name} {side} FeatureDataset is empty; required views {list(required_views)}"
        raise ValueError(msg)
    for entity_id, views in features.features.items():
        missing = [view for view in required_views if view not in views]
        if missing:
            msg = (
                f"{predictor_name} missing {side} view(s) {missing} for entity {entity_id!r}; "
                f"required={list(required_views)}"
            )
            raise ValueError(msg)


def validate_pyg_drug_graphs(
    drug_features: FeatureDataset,
    *,
    predictor_name: str,
    view: str = "drug_graph",
) -> None:
    """Validate PyG ``Data`` graphs under *view* for type, attrs, and node dims."""
    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - dependency is required for DrugGNN
        msg = f"{predictor_name} requires torch_geometric to validate drug graphs"
        raise ImportError(msg) from exc

    node_dims: set[int] = set()
    for entity_id, views in drug_features.features.items():
        graph = views.get(view)
        if graph is None:
            msg = f"{predictor_name} missing drug view {view!r} for entity {entity_id!r}"
            raise ValueError(msg)
        if not isinstance(graph, Data):
            msg = (
                f"{predictor_name} drug view {view!r} for entity {entity_id!r} "
                f"must be torch_geometric.data.Data, got {type(graph).__name__}"
            )
            raise ValueError(msg)
        if getattr(graph, "x", None) is None:
            msg = f"{predictor_name} drug graph for entity {entity_id!r} is missing attribute 'x'"
            raise ValueError(msg)
        if getattr(graph, "edge_index", None) is None:
            msg = f"{predictor_name} drug graph for entity {entity_id!r} is missing attribute 'edge_index'"
            raise ValueError(msg)
        x: Any = graph.x
        if getattr(x, "ndim", 0) != 2:
            msg = f"{predictor_name} drug graph 'x' for entity {entity_id!r} must be 2-dimensional"
            raise ValueError(msg)
        node_dims.add(int(x.shape[1]))
    if len(node_dims) > 1:
        msg = f"{predictor_name} drug graphs have inconsistent node-feature dimensions: {sorted(node_dims)}"
        raise ValueError(msg)
