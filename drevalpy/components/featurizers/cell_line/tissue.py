"""Tissue metadata featurizer for cell lines."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import TISSUE_IDENTIFIER


def _tissue_label(features: FeatureDataset, entity_id: str) -> str | None:
    views = features.features.get(str(entity_id))
    if views is None or TISSUE_IDENTIFIER not in views:
        return None
    return str(np.asarray(views[TISSUE_IDENTIFIER]).reshape(-1)[0])


@register_cell_line_featurizer(
    "tissue",
    description="One-hot encoding of tissue or lineage labels for cell-line entities.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class TissueFeaturizer(CellLineFeaturizer):
    """Map each cell line to a dense one-hot tissue vector."""

    def __init__(self, *, allow_missing: bool = False) -> None:
        self._encoder = OneHotCategoryEncoder()
        self._allow_missing = bool(allow_missing)

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> TissueFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()), dtype=str)
        available: list[str] = []
        for entity_id in ids:
            label = _tissue_label(features, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
                    raise ValueError(msg)
                continue
            available.append(label)
        if not available:
            if self._allow_missing:
                self._encoder.fit_categories(np.array([], dtype=str))
                return self
            msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
            raise ValueError(msg)
        self._encoder.fit_categories(np.asarray(available, dtype=str))
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        if self._encoder.output_dim == 0:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        categories: list[str] = []
        for entity_id in entity_ids:
            label = _tissue_label(features, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
                    raise ValueError(msg)
                categories.append("__missing__")
            else:
                categories.append(label)
        return self._encoder.transform(np.asarray(categories, dtype=str))

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "tissue": self.transform(features, entity_ids),
            "tissue_categories": np.asarray(self._encoder.categories, dtype=str),
        }

    @property
    def output_dim(self) -> int:
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        self._encoder.set_state(state)
