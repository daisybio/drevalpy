"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.log import get_logger

logger = get_logger(__name__)


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BionicCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Bionic cell line featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("bionic_features",)
    precompute: ClassVar[bool] = True

    def _compute_from_raw(
        self, source: FeatureSource, entity_ids: np.ndarray, *, gene_add_num: int = 512
    ) -> np.ndarray:
        """Compute BIONIC features on-the-fly from PPI embeddings and gene expression.

        :param source: Feature source providing cell-line views.
        :param entity_ids: Cell-line IDs.
        :param gene_add_num: Number of top-expressed genes to average.
        :returns: Float32 array of shape (len(entity_ids), embed_dim).
        """
        mdata = getattr(source, "mdata", None)
        if mdata is None or "dipk" not in mdata.uns:
            msg = "BIONIC features require DIPK data in mdata.uns['dipk']"
            raise ValueError(msg)

        dipk_data = mdata.uns["dipk"]
        ppi_features = np.asarray(dipk_data["ppi_features"])
        ppi_gene_names: list[str] = list(dipk_data["ppi_gene_names"])
        gene_list_sel: set[str] = set(dipk_data["gene_list_sel"])
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
            result[i] = self._aggregate_ppi_for_cell_line(
                expr_matrix[i], gene_names, eligible_genes, ppi_lookup, gene_add_num, embed_dim
            )
        return result

    @staticmethod
    def _aggregate_ppi_for_cell_line(
        expr_row: np.ndarray,
        gene_names: tuple[str, ...],
        eligible_genes: set[str],
        ppi_lookup: dict[str, np.ndarray],
        gene_add_num: int,
        embed_dim: int,
    ) -> np.ndarray:
        """Average PPI vectors of top-expressed eligible genes for one cell line."""
        sorted_indices = np.argsort(-expr_row)
        selected: list[np.ndarray] = []
        for idx in sorted_indices:
            if len(selected) >= gene_add_num:
                break
            gene = gene_names[idx]
            if gene in eligible_genes:
                selected.append(ppi_lookup[gene])
        if selected:
            return np.mean(selected, axis=0).astype(np.float32)
        return np.zeros(embed_dim, dtype=np.float32)

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform with fallback to on-the-fly computation.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Float32 feature matrix.
        """
        mdata = getattr(source, "mdata", None)
        matrix = self.fetch(mdata, entity_ids) if mdata is not None else None
        if matrix is not None:
            return matrix.astype(np.float32)

        try:
            matrix = source.get_view_matrix("bionic_features", entity_ids)
            if not np.all(np.isnan(matrix)):
                return matrix.astype(np.float32)
        except KeyError:
            pass

        logger.warning("bionic_features not precomputed; computing on-the-fly from PPI embeddings.")
        return self._compute_from_raw(source, entity_ids)
