"""Pathway featurizer for Precily."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import numpy as np
import pandas as pd

from drevalpy.components.core.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.log import get_logger

logger = get_logger(__name__)


@register_cell_line_featurizer(
    "pathways",
    description="GSVA pathway features computed per-split (set-dependent).",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PathwaysCellLineFeaturizer(CellLineFeaturizer):
    """Pathways cell line featurizer component.

    Set-dependent: GSVA enrichment scores depend on the distribution of all
    samples in the expression matrix, so they must be computed per-split.
    """

    input_views: ClassVar[tuple[str, ...]] = ("pathways",)
    source_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    precompute: ClassVar[bool] = False

    def __init__(self, *, view: str = "pathways") -> None:
        """Initialize instance state.

        :param view: view.
        """
        self._view = view
        self._output_dim = 0
        self._fit_scores: np.ndarray | None = None
        self._fit_ids: np.ndarray | None = None

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> PathwaysCellLineFeaturizer:
        """Compute GSVA on training cell lines.

        :param source: Feature source providing cell-line views.
        :param entity_ids: Training cell-line IDs.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        result = self._run_gsva(source, ids)
        self._fit_scores = result
        self._fit_ids = ids
        self._output_dim = int(result.shape[1])
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return GSVA scores for requested entities.

        :param source: Feature source providing cell-line views.
        :param entity_ids: Cell-line IDs to transform.
        :returns: Float32 feature matrix.
        """
        if self._fit_scores is None:
            msg = "PathwaysCellLineFeaturizer must be fit before transform"
            raise RuntimeError(msg)

        id_map = {str(id_): i for i, id_ in enumerate(self._fit_ids)}
        if all(str(id_) in id_map for id_ in entity_ids):
            indices = [id_map[str(id_)] for id_ in entity_ids]
            return self._fit_scores[indices].astype(np.float32)

        all_ids = np.unique(np.concatenate([self._fit_ids, entity_ids]))
        result = self._run_gsva(source, all_ids)
        id_map = {str(id_): i for i, id_ in enumerate(all_ids)}
        indices = [id_map[str(id_)] for id_ in entity_ids]
        return result[indices].astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing cell-line views.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=None,
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    def _run_gsva(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Run GSVA on the given cell lines.

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

        import os

        os.unlink(gmt_path)

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
