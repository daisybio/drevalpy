"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.featurizers.sample.gene_lists import checked_view_gene_lists, view_genes
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register("molirOmics", description="MOLIR multi-omics input preparation.")
class MOLIROmicsFeaturizer(Featurizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(
        self, *, n_gene_expression_features: int = 1000, gene_lists: Mapping[str, str | None] | None = None
    ) -> None:
        """Store the variance-selection feature count and the per-view gene lists.

        :param n_gene_expression_features: Number of gene-expression features to keep.
        :param gene_lists: Shipped gene-list name per view, keyed by view name; each named view must hold
            every gene of its list. A view left out or mapped to ``None`` keeps all its genes. A key that is not
        one of the three input views is rejected with ``InvalidInputError``.
        """
        self._gene_lists = checked_view_gene_lists(gene_lists, _VIEWS)
        self._n_features = int(n_gene_expression_features)
        self._scaler: StandardScaler | None = None
        self._mask: np.ndarray | None = None
        #: Names each view is read in at transform; for gene expression, before selection.
        self._view_names: dict[str, tuple[str, ...] | None] = {}
        self._selected_names: tuple[str, ...] | None = None

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
        """Return tunable variance-selection feature count.

        :returns: Hyperparameter search space mapping.
        """
        return {"n_gene_expression_features": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match the parameter-free stored variant.

        :returns: An empty mapping.
        """
        return {}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap stored gene-expression features and pass the other two views through whole.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, one per row of *matrix*.
        :param matrix: Stored gene-expression features.
        :returns: Per-omics numeric blocks.
        """
        names = {view: source.get_feature_names(view) for view in _VIEWS[1:]}
        return self._result(source, ids, matrix, None, names)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit the scaler and pick the highest-variance gene-expression features of the sorted unique samples.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        from sklearn.preprocessing import StandardScaler

        _ = train_pair_ids, es_pair_ids
        self._view_names = {view: view_genes(source, view, self._gene_lists.get(view)) for view in _VIEWS}
        ge_names = self._view_names["gene_expression"]
        matrix = np.arcsinh(view_matrix(source, "gene_expression", np.unique(train_ids), ge_names))
        self._scaler = StandardScaler().fit(matrix)
        # Standardization makes every nonconstant variance approximately one;
        # rank expression variance before that step, using training rows only.
        variances = np.var(matrix, axis=0)
        mask = np.zeros(len(variances), dtype=bool)
        mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True
        self._mask = mask
        self._selected_names = None if ge_names is None else tuple(np.array(ge_names)[mask])
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the scaled, selected gene expression and the two other omics views untouched.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: Per-omics numeric blocks, projected as the gene-expression block.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._scaler is None or self._mask is None:
            msg = "MOLIROmicsFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            matrix = np.empty((0, int(self._mask.sum())))
        else:
            raw = view_matrix(source, "gene_expression", ids, self._view_names.get("gene_expression"))
            matrix = self._scaler.transform(np.arcsinh(raw))[:, self._mask]
        return self._result(source, ids, matrix, self._selected_names, self._view_names)

    def _result(
        self,
        source: FeatureSource,
        ids: np.ndarray,
        gene_expression: np.ndarray,
        gene_expression_names: tuple[str, ...] | None,
        view_names: Mapping[str, tuple[str, ...] | None],
    ) -> FeaturizerResult:
        """Assemble the gene-expression block and the pass-through blocks of the two other views.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :param gene_expression: Gene-expression features, one row per ID.
        :param gene_expression_names: Column names of *gene_expression*.
        :param view_names: Names the two other views are read in.
        :returns: Per-omics numeric blocks, projected as the gene-expression block.
        """
        blocks = {
            "gene_expression": numeric_feature_block(
                gene_expression.astype(np.float32), feature_names=gene_expression_names
            ),
            **{
                view: numeric_feature_block(
                    view_matrix(source, view, ids, view_names.get(view)).astype(np.float32),
                    feature_names=view_names.get(view),
                )
                for view in _VIEWS[1:]
            },
        }
        return FeaturizerResult(ids, blocks, ("gene_expression",))
