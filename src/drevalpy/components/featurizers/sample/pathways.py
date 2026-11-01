"""Pathway featurizer for Precily."""

from __future__ import annotations

import os
import tempfile
from typing import ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.data._msigdb import get_precily_gmt
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side
from drevalpy.types.data.views import PATHWAYS_GMT_KEY


@register("pathways", description="GSVA pathway features computed per-split (set-dependent).")
class PathwaysSampleFeaturizer(Featurizer):
    """Pathways sample featurizer component.

    Set-dependent: GSVA enrichment scores depend on the distribution of all
    samples in the expression matrix, so they must be computed per-split.

    The gene sets come from ``mdata.uns["pathways_gmt"]`` when the dataset carries them, and
    otherwise from the MSigDB C2 CP v6.1 file that ``drevalpy.data._msigdb.get_precily_gmt``
    downloads into the cache on first use.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = ("pathways",)
    source_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self, *, view: str = "pathways") -> None:
        """Bind the name of the emitted pathway view.

        :param view: Block name for the pathway scores; also the view name reported as input.
        """
        self._view = view
        self._fit_scores: np.ndarray | None = None
        self._fit_ids: np.ndarray | None = None
        self._pathway_names: tuple[str, ...] | None = None
        self._gmt_text: str | None = None

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Compute GSVA on the training samples, in the given order, and keep the scores as reference.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = train_pair_ids, es_pair_ids
        self._pathway_names = None
        self._gmt_text = None
        self._fit_scores = self._run_gsva(source, train_ids)
        self._fit_ids = np.asarray(train_ids)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return GSVA scores for *ids*: the fitted ones, re-running GSVA with the training samples for others.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block named after the view, labelled with the pathway names.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._fit_scores is None or self._fit_ids is None:
            msg = "PathwaysSampleFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        id_map = {str(id_): i for i, id_ in enumerate(self._fit_ids)}
        if all(str(id_) in id_map for id_ in ids):
            matrix = self._fit_scores[[id_map[str(id_)] for id_ in ids]]
        else:
            all_ids = np.unique(np.concatenate([self._fit_ids, ids]))
            result = self._run_gsva(source, all_ids)
            new_map = {str(id_): i for i, id_ in enumerate(all_ids)}
            matrix = np.asarray(
                [
                    self._fit_scores[id_map[str(id_)]] if str(id_) in id_map else result[new_map[str(id_)]]
                    for id_ in ids
                ],
                dtype=np.float32,
            )
        return dense_result(ids, self._view, matrix, self._pathway_names)

    def _run_gsva(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Run GSVA on the given samples.

        :param source: Feature source providing sample views.
        :param entity_ids: Sample IDs.
        :returns: Float32 array of shape (len(entity_ids), n_pathways).
        :raises ValueError: If the gene expression view has no feature names, or the GSVA output
            lacks pathways from the fitted schema.
        """
        import gseapy as gp

        if self._gmt_text is None:
            stored = source.uns(PATHWAYS_GMT_KEY)
            self._gmt_text = get_precily_gmt().read_text(encoding="utf-8") if stored is None else str(stored)

        import pandas as pd

        expr_matrix = source.view("gene_expression", entity_ids)
        gene_names = source.feature_names("gene_expression")
        if gene_names is None:
            msg = "gene_expression view must provide feature names"
            raise ValueError(msg)

        expr_df = pd.DataFrame(expr_matrix, index=entity_ids, columns=list(gene_names))
        expr_df = expr_df.loc[~expr_df.index.duplicated(keep="first")]
        expr_genes_by_samples = expr_df.T

        with tempfile.TemporaryDirectory() as gmt_dir:
            gmt_path = os.path.join(gmt_dir, "pathways.gmt")
            with open(gmt_path, "w", encoding="utf-8") as f:
                f.write(self._gmt_text)
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
        if self._pathway_names is None:
            self._pathway_names = tuple(scores.columns)
        elif not set(self._pathway_names).issubset(scores.columns):
            raise ValueError("GSVA output is missing pathways from the fitted schema")
        scores = scores.loc[:, list(self._pathway_names)]

        result = np.zeros((len(entity_ids), scores.shape[1]), dtype=np.float32)
        for i, cl in enumerate(entity_ids):
            if cl in scores.index:
                result[i] = scores.loc[cl].values

        return result
