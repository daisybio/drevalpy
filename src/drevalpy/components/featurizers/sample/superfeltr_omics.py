"""SuperFELTR multi-omics feature-selection featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.featurizers.sample.gene_lists import checked_view_gene_lists, view_genes
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register("superfeltrOmics", side=Side.SAMPLE, description="SuperFELTR variance-selected multi-omics inputs.")
class SuperFELTROmicsFeaturizer(Featurizer):
    """Select high-variance features independently in each SuperFELTR view."""

    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(self, *, n_features_per_view: int = 1000, gene_lists: Mapping[str, str | None] | None = None) -> None:
        """Store the per-view variance-selection feature count and gene lists.

        :param n_features_per_view: Number of features to keep per omics view.
        :param gene_lists: Shipped gene-list name per view, keyed by view name; each named view must hold
            every gene of its list. A view left out or mapped to ``None`` keeps all its genes. A key that is not
        one of the three input views is rejected with ``InvalidInputError``.
        """
        self._gene_lists = checked_view_gene_lists(gene_lists, _VIEWS)
        self._n_features = int(n_features_per_view)
        self._masks: dict[str, np.ndarray] | None = None
        self._feature_names: dict[str, tuple[str, ...]] = {}
        self._source_names: dict[str, tuple[str, ...] | None] = {}

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Declare one block per omics view.

        :param config: Unused.
        :returns: The ``gene_expression``, ``mutations`` and ``copy_number_variation_gistic`` blocks.
        """
        _ = config
        return tuple(BlockSpec(view, FeatureFormat.NUMERIC_MATRIX) for view in _VIEWS)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable per-view variance-selection feature count.

        :returns: Hyperparameter search space mapping.
        """
        return {"n_features_per_view": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit variance selectors independently in each omics view on the sorted unique training samples.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = train_pair_ids, es_pair_ids
        ids = np.unique(train_ids)
        masks: dict[str, np.ndarray] = {}
        for view in _VIEWS:
            self._source_names[view] = view_genes(source, view, self._gene_lists.get(view))
            variances = np.var(self._view_matrix(source, view, ids), axis=0)
            mask = np.zeros(len(variances), dtype=bool)
            mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True
            masks[view] = mask
            names = self._source_names[view]
            self._feature_names[view] = () if names is None else tuple(np.array(names)[mask])
        self._masks = masks
        return self

    def _view_matrix(self, source: FeatureSource, view: str, ids: np.ndarray) -> np.ndarray:
        """Read *view* in fitted column order and apply the expression transform before selection."""
        matrix = view_matrix(source, view, ids, self._source_names.get(view))
        return np.arcsinh(matrix) if view == "gene_expression" else matrix

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return per-omics numeric blocks with variance-selected columns.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block per omics view, projected as the gene-expression block.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._masks is None:
            msg = "SuperFELTROmicsFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        blocks = {
            view: numeric_feature_block(
                self._view_matrix(source, view, ids)[:, mask].astype(np.float32),
                feature_names=self._feature_names.get(view),
            )
            for view, mask in self._masks.items()
        }
        return FeaturizerResult(ids, blocks, ("gene_expression",))
