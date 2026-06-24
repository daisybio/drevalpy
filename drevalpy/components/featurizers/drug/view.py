"""Drug featurizers for drug features."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "view",
    description="Pass through one dense drug view from a FeatureDataset.",
    category="native",
)
class ViewDrugFeaturizer(DrugFeaturizer):
    """Featurize one drug view without additional transformation."""

    def __init__(self, *, view: str = "fingerprints") -> None:
        self._view = view
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> ViewDrugFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim


@register_drug_featurizer(
    "fingerprints",
    description="Precomputed Morgan fingerprints loaded from the fingerprints view.",
    category="general_purpose",
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Alias for the standard fingerprints view."""

    def __init__(self, *, view: str = "fingerprints", n_bits: int = 128) -> None:
        super().__init__(view=view)
        self._n_bits = int(n_bits)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_bits": {
                "type": "categorical",
                "choices": [128, 256, 512],
                "default": 128,
            },
        }


@register_drug_featurizer(
    "oneHot",
    description="One-hot encoding of drug identifiers.",
    category="native",
)
class OneHotDrugFeaturizer(DrugFeaturizer):
    """Fit a one-hot space over all drugs present in the feature dataset."""

    def __init__(self) -> None:
        self._drug_to_index: dict[str, int] = {}
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> OneHotDrugFeaturizer:
        drug_ids = sorted(features.features.keys())
        self._drug_to_index = {str(drug_id): index for index, drug_id in enumerate(drug_ids)}
        self._output_dim = len(self._drug_to_index)
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(entity_ids), self._output_dim), dtype=np.float32)
        for row, drug_id in enumerate(entity_ids):
            index = self._drug_to_index.get(str(drug_id))
            if index is None:
                msg = f"Unknown drug id {drug_id!r} for oneHot featurizer"
                raise KeyError(msg)
            matrix[row, index] = 1.0
        return matrix

    @property
    def output_dim(self) -> int:
        return self._output_dim


@register_drug_featurizer(
    "drugGraph",
    description="Precomputed PyG molecular graphs stored under the drug_graph view.",
    category="general_purpose",
)
class DrugGraphFeaturizer(DrugFeaturizer):
    """Expose precomputed drug graphs for graph predictors."""

    output_contract = FeatureContract(
        kind=FeatureKind.GRAPH,
        view="drug_graph",
        backend="pyg",
        scope="per_drug",
        has_node_features=True,
        has_edge_features=True,
    )

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
