"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.data.artifacts import get_artifact
from drevalpy.log import get_logger
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource

logger = get_logger(__name__)


@register(
    "bionic",
    description="BIONIC PPI-based sample features for DIPK.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class BionicSampleFeaturizer(DenseViewSampleFeaturizer):
    """Bionic sample featurizer component.

    Uses pre-trained BIONIC gene embeddings (from PPI networks) to create
    per-sample feature vectors by aggregating the PPI embeddings of the
    top-expressed genes.
    """

    input_views: ClassVar[tuple[str, ...]] = ("bionic_features",)
    source_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str | None = None, gene_add_num: int = 256, aggregation: str = "mean") -> None:
        """Configure the gene aggregation of the on-the-fly BIONIC features.

        :param view: Name of the view holding pre-computed features; ``None`` uses the declared input view.
        :param gene_add_num: Number of top-expressed genes to aggregate.
        :param aggregation: Aggregation method for gene embeddings ("mean", "max", "sum").
        """
        super().__init__(view=view)
        self._gene_add_num = int(gene_add_num)
        self._aggregation = aggregation

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "gene_add_num": {"type": "categorical", "choices": [128, 256, 512, 1024], "default": 256},
            "aggregation": {"type": "categorical", "choices": ["mean", "max", "sum"], "default": "mean"},
        }

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
