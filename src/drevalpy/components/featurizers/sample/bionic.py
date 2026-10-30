"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.data.artifacts import get_artifact
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side


@register("bionic", description="BIONIC PPI-based sample features for DIPK.")
class BionicSampleFeaturizer(Featurizer):
    """Bionic sample featurizer component.

    Uses pre-trained BIONIC gene embeddings (from PPI networks) to create
    per-sample feature vectors by aggregating the PPI embeddings of the
    top-expressed genes. A source that holds the view already passes it through.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = ("bionic_features",)
    source_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self, *, view: str | None = None, gene_add_num: int = 256, aggregation: str = "mean") -> None:
        """Configure the gene aggregation of the on-the-fly BIONIC features.

        :param view: Name of the view holding pre-computed features; ``None`` uses the declared input view.
        :param gene_add_num: Number of top-expressed genes to aggregate.
        :param aggregation: Aggregation method for gene embeddings ("mean", "max", "sum").
        """
        self._view = view or self.input_views[0]
        self._gene_add_num = int(gene_add_num)
        self._aggregation = aggregation
        self._names: tuple[str, ...] | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "gene_add_num": {"type": "categorical", "choices": [128, 256, 512, 1024], "default": 256},
            "aggregation": {"type": "categorical", "choices": ["mean", "max", "sum"], "default": "mean"},
        }

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID; all ``True`` when the view is absent.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match the parameter-free stored variant.

        :returns: An empty mapping.
        """
        return {}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap stored BIONIC features as the view's block.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, one per row of *matrix*.
        :param matrix: Stored BIONIC features.
        :returns: One block named after the view.
        """
        names = source.get_feature_names(self._view) if source.has_view(self._view) else None
        return dense_result(ids, self._view, matrix, names)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Record the view's feature names when the source holds the view; nothing is learned.

        :param source: Feature source of the sample side.
        :param train_ids: Unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = train_ids, train_pair_ids, es_pair_ids
        self._names = source.get_feature_names(self._view) if source.has_view(self._view) else None
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the view's rows for *ids*, or compute them from gene expression when the view is absent.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block named after the view.
        """
        if source.has_view(self._view):
            names = self._names or source.get_feature_names(self._view)
            return dense_result(ids, self._view, view_matrix(source, self._view, ids, names), names)
        return dense_result(ids, self._view, self._compute_from_source(source, ids), None)

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute BIONIC features from gene expression + downloaded PPI embeddings.

        :param source: Feature source providing sample views.
        :param entity_ids: Sample IDs.
        :returns: Float32 array of shape (len(entity_ids), embed_dim).
        :raises ValueError: If the gene expression view has no feature names.
        """
        ppi_features, ppi_gene_names, gene_list_sel = _load_ppi_data()

        ppi_lookup = {gene: ppi_features[i] for i, gene in enumerate(ppi_gene_names)}
        eligible_genes = gene_list_sel & set(ppi_gene_names)

        expr_matrix = source.get_view_matrix("gene_expression", entity_ids)
        gene_names = source.get_feature_names("gene_expression")
        if gene_names is None:
            msg = "gene_expression view must provide feature names"
            raise ValueError(msg)

        embed_dim = ppi_features.shape[1]
        result = np.zeros((len(entity_ids), embed_dim), dtype=np.float32)
        for i in range(len(entity_ids)):
            result[i] = _aggregate_ppi_for_sample(
                expr_matrix[i],
                gene_names,
                eligible_genes,
                ppi_lookup,
                self._gene_add_num,
                embed_dim,
                self._aggregation,
            )
        return result


def _load_ppi_data() -> tuple[np.ndarray, list[str], set[str]]:
    """Load PPI features and gene list from the artifact cache (auto-downloads if needed).

    :returns: Tuple of (float64 ppi_features array, gene_names list, gene_list_sel set).
    """
    import pandas as pd

    ppi_path = get_artifact("human_ppi_features.tsv")
    gene_list_path = get_artifact("gene_list_sel.txt")

    ppi_df = pd.read_csv(ppi_path, index_col=0, sep="\t")
    ppi_features = ppi_df.to_numpy(dtype=np.float64)
    ppi_gene_names = list(ppi_df.index)

    with open(gene_list_path, encoding="utf-8") as f:
        gene_list_sel = {line.strip() for line in f if line.strip()}

    return ppi_features, ppi_gene_names, gene_list_sel


def _aggregate_ppi_for_sample(
    expr_row: np.ndarray,
    gene_names: tuple[str, ...],
    eligible_genes: set[str],
    ppi_lookup: dict[str, np.ndarray],
    gene_add_num: int,
    embed_dim: int,
    aggregation: str,
) -> np.ndarray:
    """Aggregate PPI vectors of top-expressed eligible genes for one sample.

    Ties are broken by column order (stable sort), like the stable ``sorted`` of drevalpy 1.6.1, and
    the PPI vectors are aggregated in float64 before the cast to float32.
    """
    sorted_indices = np.argsort(-expr_row, kind="stable")
    selected: list[np.ndarray] = []
    for idx in sorted_indices:
        if len(selected) >= gene_add_num:
            break
        gene = gene_names[idx]
        if gene in eligible_genes:
            selected.append(ppi_lookup[gene])
    if not selected:
        return np.zeros(embed_dim, dtype=np.float32)
    stacked = np.array(selected)
    if aggregation == "mean":
        return stacked.mean(axis=0).astype(np.float32)
    if aggregation == "max":
        return stacked.max(axis=0).astype(np.float32)
    if aggregation == "sum":
        return stacked.sum(axis=0).astype(np.float32)
    return stacked.mean(axis=0).astype(np.float32)
