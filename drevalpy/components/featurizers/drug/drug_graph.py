"""Precomputed molecular graph drug featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "drugGraph",
    description="Precomputed PyG molecular graphs stored under the drug_graph view.",
    category="general_purpose",
    contract=FeatureKind.GRAPH,
)
class DrugGraphFeaturizer(DrugFeaturizer):
    """Expose precomputed drug graphs for graph predictors."""

    def __init__(self, *, view: str = "drug_graph") -> None:
        self._view = view
        self._graphs: dict[str, object] = {}
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> DrugGraphFeaturizer:
        for drug_id, views in features.features.items():
            if self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_id!r}"
                raise KeyError(msg)
            self._graphs[str(drug_id)] = views[self._view]
        if self._graphs:
            first = next(iter(self._graphs.values()))
            self._output_dim = int(getattr(first, "num_node_features", 0))
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        graphs: list[object] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._graphs:
                graphs.append(self._graphs[drug_key])
                continue
            views = features.features.get(drug_key)
            if views is None or self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_key!r}"
                raise KeyError(msg)
            graphs.append(views[self._view])
        return np.array(graphs, dtype=object)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def graph_by_drug(self) -> dict[str, object]:
        return self._graphs
