"""Cell-line featurizer example: standardize one raw omics view.

Shows the three hooks every featurizer implements -- ``_fit``,
``_transform_blocks`` and the ``output_dim`` property -- plus the ``input_views``
declaration, without which registration is rejected.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.plugin import (
    CellLineFeaturizer,
    FeatureBlock,
    FeatureFormat,
    FeatureSource,
    numeric_feature_block,
    register_cell_line_featurizer,
)


@register_cell_line_featurizer(
    "toyCellLine",
    description="Gene expression standardized with statistics learned on the training cell lines.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyCellLineFeaturizer(CellLineFeaturizer):
    """Standardize the ``gene_expression`` view column by column."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self) -> None:
        """Create an unfitted featurizer."""
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> ToyCellLineFeaturizer:
        """Learn one mean and one standard deviation per column."""
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        matrix = source.get_view_matrix(self.input_views[0], ids)
        self._mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
        scale = np.asarray(matrix.std(axis=0), dtype=np.float64)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        return self

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return the standardized view as one named numeric block.

        The block name matters: a ``BlockPredictor`` asks for it by name through
        ``required_cell_line_blocks``. Naming it after the input view is the
        default the registry assumes when a featurizer declares no
        ``output_block_specs``.
        """
        if self._mean is None or self._scale is None:
            msg = "ToyCellLineFeaturizer must be fitted before transforming"
            raise RuntimeError(msg)
        matrix = source.get_view_matrix(self.input_views[0], entity_ids)
        standardized = ((matrix - self._mean) / self._scale).astype(np.float32)
        return {self.input_views[0]: numeric_feature_block(standardized)}

    @property
    def output_dim(self) -> int:
        """Number of feature columns produced after fitting."""
        return 0 if self._mean is None else int(self._mean.shape[0])

    def get_state(self) -> dict[str, object]:
        """Return the fitted statistics so a checkpoint can round-trip them."""
        if self._mean is None or self._scale is None:
            return {}
        return {"mean": self._mean.tolist(), "scale": self._scale.tolist()}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the statistics a previous ``get_state`` returned."""
        mean = state.get("mean")
        scale = state.get("scale")
        if mean is None or scale is None:
            return
        self._mean = np.asarray(mean, dtype=np.float64)
        self._scale = np.asarray(scale, dtype=np.float64)
