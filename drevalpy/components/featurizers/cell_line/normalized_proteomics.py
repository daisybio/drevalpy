"""Normalized proteomics featurizer for cell lines."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.types.data.batch.feature_block import FeatureBlock, numeric_feature_block


def log10_and_set_na(x: np.ndarray) -> np.ndarray:
    """Log10 transform and set NaN for infinite values.

    :param x: input array
    :returns: log10 transformed array with NaN for infinite values
    """
    x = np.log10(x)
    x[np.isinf(x)] = np.nan
    return x


class ProteomicsMedianCenterAndImputeTransformer(BaseEstimator, TransformerMixin):
    """Performs median centering and imputation of proteomics data."""

    def __init__(
        self,
        feature_threshold=0.7,
        n_features=1000,
        normalization_downshift=1.8,
        normalization_width=0.3,
        imputation_seed=100,
    ):
        """Hyperparameters for the normalization.

        :param feature_threshold: Minimum fraction of non-missing protein values per feature.
        :param n_features: Fallback feature count when thresholding leaves too few features.
        :param normalization_downshift: Downshift factor for the mean.
        :param normalization_width: Width factor for the standard deviation.
        :param imputation_seed: Seed for per-call missing-value imputation without touching the
            global NumPy RNG state.
        """
        self.feature_threshold = feature_threshold
        self.n_features = n_features
        self.normalization_downshift = normalization_downshift
        self.normalization_width = normalization_width
        self.imputation_seed = imputation_seed
        self.protein_indices = np.array([])
        self.mean_median = 0

    def fit(self, X, y=None):  # noqa: N803  # sklearn API
        """Learn top n_feature complete proteins and calculate the mean median of train cell lines.

        :param X: input proteomics data
        :param y: not used
        :returns: self
        """
        required_proteins = int(X.shape[0] * self.feature_threshold)
        completeness = np.sum(~np.isnan(X), axis=0)
        n_complete_features = np.count_nonzero(completeness >= required_proteins)
        if n_complete_features < self.n_features:
            sorted_indices = np.argsort(completeness)[::-1]
            self.protein_indices = sorted_indices[: self.n_features]
        else:
            self.protein_indices = np.where(completeness >= required_proteins)[0]
        selected_proteins = X[:, self.protein_indices]
        medians = np.nanmedian(selected_proteins, axis=1)
        self.mean_median = np.nanmean(medians)
        return self

    def transform(self, X):  # noqa: N803  # sklearn API
        """Median center the data and impute missing values with downshifted normal distribution.

        :param X: input proteomics data
        :returns: transformed proteomics data
        """
        proteomics_vector = X[0][self.protein_indices]

        correction_factor = self.mean_median / np.nanmedian(proteomics_vector)
        proteomics_vector = proteomics_vector * correction_factor
        cell_line_mean = np.nanmean(proteomics_vector)
        cell_line_sd = np.nanstd(proteomics_vector)
        downshifted_mean = cell_line_mean - (self.normalization_downshift * cell_line_sd)
        shrinked_sd = self.normalization_width * cell_line_sd
        n_missing = np.count_nonzero(np.isnan(proteomics_vector))
        rng = np.random.default_rng(self.imputation_seed)
        proteomics_vector[np.isnan(proteomics_vector)] = rng.normal(
            loc=downshifted_mean, scale=shrinked_sd, size=n_missing
        )
        return [proteomics_vector]


@register_cell_line_featurizer(
    "normalizedProteomics",
    description="Proteomics view with log10 transform, median centering, and imputation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class NormalizedProteomicsCellLineFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("proteomics",)

    def __init__(
        self,
        *,
        view: str = "proteomics",
        proteomics_feature_threshold: float = 0.7,
        proteomics_n_features: int = 1000,
        proteomics_normalization_width: float = 0.3,
        proteomics_normalization_downshift: float = 1.8,
    ) -> None:
        """Initialize instance state.

        :param view: view.
        :param proteomics_feature_threshold: proteomics feature threshold.
        :param proteomics_n_features: proteomics n features.
        :param proteomics_normalization_width: proteomics normalization width.
        :param proteomics_normalization_downshift: proteomics normalization downshift.
        """
        self._view = view
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )
        self._output_dim = 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "proteomics_feature_threshold": {"type": "float", "low": 0.3, "high": 0.9, "default": 0.7},
            "proteomics_n_features": {"type": "int", "low": 500, "high": 2000, "default": 1000},
            "proteomics_normalization_downshift": {"type": "float", "low": 1.0, "high": 3.0, "default": 1.8},
            "proteomics_normalization_width": {"type": "float", "low": 0.1, "high": 0.6, "default": 0.3},
        }

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> NormalizedProteomicsCellLineFeaturizer:
        """Fit on training data.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, ids) if mdata is not None else None
        if precomputed is not None:
            self._output_dim = int(precomputed.shape[1])
            return self
        matrix = log10_and_set_na(source.get_view_matrix(self._view, np.unique(ids)))
        self._transformer.fit(matrix)
        self._output_dim = len(self._transformer.protein_indices)
        return self

    def _transform_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Get log10-transformed matrix and apply the fitted transformer row-by-row."""
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        matrix = log10_and_set_na(source.get_view_matrix(self._view, entity_ids))
        rows = []
        for row in matrix:
            rows.append(self._transformer.transform(row[None, :])[0])
        return np.vstack(rows).astype(np.float32)

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return self._transform_matrix(source, entity_ids)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {
            "proteomics_transformer": self._transformer,
            "view": self._view,
            "output_dim": self._output_dim,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        transformer = state.get("proteomics_transformer")
        if isinstance(transformer, ProteomicsMedianCenterAndImputeTransformer):
            self._transformer = transformer
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
