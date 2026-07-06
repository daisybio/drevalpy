"""MolGNet drug featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "molgnet",
    description="Precomputed MolGNet drug embeddings for DIPK.",
    category="general_purpose",
    contract=FeatureKind.DENSE,
)
class MolGNetDrugFeaturizer(DrugFeaturizer):
    """Expose variable-size MolGNet tensors without stacking into one dense matrix."""

    def __init__(self, *, view: str = "molgnet_features") -> None:
        self._view = view
        self._features_by_drug: dict[str, np.ndarray] = {}
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> MolGNetDrugFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        self._features_by_drug = {}
        for drug_id in ids:
            views = features.features[str(drug_id)]
            if self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_id!r}"
                raise KeyError(msg)
            self._features_by_drug[str(drug_id)] = np.asarray(views[self._view])
        if self._features_by_drug:
            first = next(iter(self._features_by_drug.values()))
            self._output_dim = int(first.shape[1]) if first.ndim == 2 else int(first.size)
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._features_by_drug:
                rows.append(self._features_by_drug[drug_key])
                continue
            views = features.features.get(drug_key)
            if views is None or self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_key!r}"
                raise KeyError(msg)
            rows.append(np.asarray(views[self._view]))
        return np.array(rows, dtype=object)

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, np.ndarray]:
        return {self._view: self.transform(features, entity_ids)}

    @property
    def output_dim(self) -> int:
        return self._output_dim
