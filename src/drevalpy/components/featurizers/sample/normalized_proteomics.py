"""Normalized proteomics featurizer for samples.

``ProteomicsMedianCenterAndImputeTransformer`` lives in
``_proteomics_transformer`` because it subclasses ``sklearn.base.BaseEstimator``,
which cannot be deferred past the ``class`` statement, so importing it here at
registration time would import ``sklearn``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from drevalpy.components.featurizers.sample._proteomics_transformer import (
        ProteomicsMedianCenterAndImputeTransformer,
    )


def log10_and_set_na(x: np.ndarray) -> np.ndarray:
    """Log10 transform and set NaN for infinite values.

    :param x: Input array; modified in place.
    :returns: ``log10`` of *x* with NaN where the result is infinite
    """
    x = np.log10(x)
    x[np.isinf(x)] = np.nan
    return x


@register(
    "normalizedProteomics",
    side=Side.SAMPLE,
    description="Proteomics view with log10 transform, median centering, and imputation.",
)
class NormalizedProteomicsSampleFeaturizer(Featurizer):
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
        """Bind the proteomics view and configure the normalizing transformer.

        :param view: Name of the proteomics view to read.
        :param proteomics_feature_threshold: Minimum fraction of non-missing values a protein needs
            to be kept.
        :param proteomics_n_features: Number of proteins kept when thresholding leaves fewer.
        :param proteomics_normalization_width: Width factor for the standard deviation of the
            imputation distribution.
        :param proteomics_normalization_downshift: Downshift of the imputation distribution's mean,
            in standard deviations.
        """
        self._view = view
        self._settings = {
            "feature_threshold": proteomics_feature_threshold,
            "n_features": proteomics_n_features,
            "normalization_width": proteomics_normalization_width,
            "normalization_downshift": proteomics_normalization_downshift,
        }
        self._names: tuple[str, ...] | None = None
        self._transformer: ProteomicsMedianCenterAndImputeTransformer | None = None

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

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit the median-centering transformer on the log10 rows of the sorted unique training samples.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        from drevalpy.components.featurizers.sample._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer,
        )

        _ = train_pair_ids, es_pair_ids
        self._names = source.feature_names(self._view)
        matrix = log10_and_set_na(view_matrix(source, self._view, np.unique(train_ids), self._names))
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(**self._settings).fit(matrix)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Log10-transform the rows of *ids* and normalize them one sample at a time.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block of the retained proteins, named after the view.
        :raises RuntimeError: If called before ``fit``.
        """
        transformer = self._transformer
        if transformer is None:
            msg = "NormalizedProteomicsSampleFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        raw = log10_and_set_na(view_matrix(source, self._view, ids, self._names))
        rows = [transformer.transform(row[None, :])[0] for row in raw]
        matrix = np.vstack(rows) if rows else np.empty((0, len(transformer.protein_indices)))
        names = None if self._names is None else tuple(self._names[i] for i in transformer.protein_indices)
        return dense_result(ids, self._view, matrix, names)
