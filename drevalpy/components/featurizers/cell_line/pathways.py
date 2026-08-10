"""Pathway featurizer for Precily."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import numpy as np
import pandas as pd

from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.log import get_logger

logger = get_logger(__name__)


@register_cell_line_featurizer(
    "pathways",
    description="Precomputed GSVA pathway features for Precily.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PathwaysCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Pathways cell line featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("pathways",)
    precompute: ClassVar[bool] = True

    def _compute_from_raw(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute GSVA pathway features on-the-fly from gene expression.

        :param source: Feature source providing cell-line views.
        :param entity_ids: Cell-line IDs.
        :returns: Float32 array of shape (len(entity_ids), n_pathways).
        """
        import gseapy as gp

        mdata = getattr(source, "mdata", None)
        if mdata is None or "pathways_gmt" not in mdata.uns:
            msg = "Pathway features require pathways_gmt in mdata.uns"
            raise ValueError(msg)

        expr_matrix = source.get_view_matrix("gene_expression", entity_ids)
        gene_names = source.get_feature_names("gene_expression")
        if gene_names is None:
            msg = "gene_expression view must provide feature names"
            raise ValueError(msg)

        expr_df = pd.DataFrame(expr_matrix, index=entity_ids, columns=list(gene_names))
        expr_df = expr_df.loc[~expr_df.index.duplicated(keep="first")]
        expr_genes_by_samples = expr_df.T

        gmt_text: str = mdata.uns["pathways_gmt"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gmt", delete=False) as f:
            f.write(gmt_text)
            gmt_path = f.name

        gv = gp.gsva(
            data=expr_genes_by_samples,
            gene_sets=gmt_path,
            kcdf="Gaussian",
            min_size=5,
            max_size=2000,
            mx_diff=True,
            threads=4,
            seed=42,
            outdir=None,
            verbose=False,
        )

        long = gv.res2d.copy()
        cols = {c.lower(): c for c in long.columns}
        term_col = cols.get("term", "Term")
        name_col = cols.get("name", "Name")
        es_col = cols.get("es", cols.get("nes", "ES"))
        wide = long.pivot(index=term_col, columns=name_col, values=es_col)
        scores = wide.T.astype(np.float32)

        result = np.zeros((len(entity_ids), scores.shape[1]), dtype=np.float32)
        for i, cl in enumerate(entity_ids):
            if cl in scores.index:
                result[i] = scores.loc[cl].values

        return result

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
            matrix = source.get_view_matrix("pathways", entity_ids)
            if not np.all(np.isnan(matrix)):
                return matrix.astype(np.float32)
        except KeyError:
            pass

        logger.warning("pathways view not precomputed; computing GSVA on-the-fly.")
        return self._compute_from_raw(source, entity_ids)
