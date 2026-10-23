"""Pathway featurizer for Precily."""

from __future__ import annotations

import os
import tempfile
from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.log import get_logger
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

logger = get_logger(__name__)


@register(
    "pathways",
    description="GSVA pathway features computed per-split (set-dependent).",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PathwaysSampleFeaturizer(SampleFeaturizer):
    """Pathways sample featurizer component.

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
        self._pathway_names: tuple[str, ...] | None = None
        self._gmt_text: str | None = None

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> PathwaysSampleFeaturizer:
        """Compute GSVA on training samples.

        :param source: Feature source providing sample views.
        :param entity_ids: Training sample IDs.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._pathway_names = None
        self._gmt_text = None
        result = self._run_gsva(source, ids)
        self._fit_scores = result
        self._fit_ids = ids
        self._output_dim = int(result.shape[1])
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return GSVA scores for requested entities.

        :param source: Feature source providing sample views.
        :param entity_ids: Sample IDs to transform.
        :returns: Float32 feature matrix.
        """
        if self._fit_scores is None:
            msg = "PathwaysSampleFeaturizer must be fit before transform"
            raise RuntimeError(msg)

        id_map = {str(id_): i for i, id_ in enumerate(self._fit_ids)}
        if all(str(id_) in id_map for id_ in entity_ids):
            indices = [id_map[str(id_)] for id_ in entity_ids]
            return self._fit_scores[indices].astype(np.float32)

        all_ids = np.unique(np.concatenate([self._fit_ids, entity_ids]))
        result = self._run_gsva(source, all_ids)
        new_map = {str(id_): i for i, id_ in enumerate(all_ids)}
        return np.asarray(
            [
                self._fit_scores[id_map[str(id_)]] if str(id_) in id_map else result[new_map[str(id_)]]
                for id_ in entity_ids
            ],
            dtype=np.float32,
        )

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Transform blocks.

        :param source: Feature source providing sample views.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        matrix = self._feature_matrix(source, entity_ids)
        return FeaturizerResult(
            entity_ids,
            {
                self._view: numeric_feature_block(
                    matrix,
                    feature_names=self._pathway_names,
                )
            },
            (self._view,),
        )

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        """Persist the training-reference scores and entity order.

        :returns: Fitted GSVA state.
        """
        return {
            "fit_scores": self._fit_scores,
            "fit_ids": self._fit_ids,
            "output_dim": self._output_dim,
            "pathway_names": self._pathway_names,
            "gmt_text": self._gmt_text,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the training reference for subsequent transformations.

        :param state: Mapping produced by get_state.
        """
        scores, ids = state.get("fit_scores"), state.get("fit_ids")
        if not isinstance(scores, np.ndarray) or not isinstance(ids, np.ndarray):
            raise ValueError("Pathways checkpoint requires fitted score and identifier arrays")
        if scores.ndim != 2 or ids.ndim != 1 or scores.shape != (len(ids), state.get("output_dim")):
            raise ValueError("Pathways checkpoint has inconsistent fitted dimensions")
        self._fit_scores, self._fit_ids = scores.copy(), ids.copy()
        self._output_dim = scores.shape[1]
        names, gmt = state.get("pathway_names"), state.get("gmt_text")
        if names is not None and (not isinstance(names, (list, tuple)) or len(names) != self._output_dim):
            raise ValueError("Pathways checkpoint has inconsistent pathway names")
        self._pathway_names = tuple(names) if names is not None else None
        self._gmt_text = gmt if isinstance(gmt, str) else None

    def _run_gsva(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Run GSVA on the given samples.

        :param source: Feature source providing sample views.
        :param entity_ids: Sample IDs.
        :returns: Float32 array of shape (len(entity_ids), n_pathways).
        """
        import gseapy as gp

        mdata = getattr(source, "mdata", None)
        if self._gmt_text is None:
            if mdata is None or "pathways_gmt" not in mdata.uns:
                msg = "Pathway features require pathways_gmt in mdata.uns"
                raise ValueError(msg)
            self._gmt_text = str(mdata.uns["pathways_gmt"])

        import pandas as pd

        expr_matrix = source.get_view_matrix("gene_expression", entity_ids)
        gene_names = source.get_feature_names("gene_expression")
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
