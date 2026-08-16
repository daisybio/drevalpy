"""Normalized proteomics featurizer for cell lines.

``ProteomicsMedianCenterAndImputeTransformer`` lives in
``_proteomics_transformer`` because it subclasses ``sklearn.base.BaseEstimator``,
which cannot be deferred past the ``class`` statement. It is re-exported here
through ``__getattr__`` so the historical import path - and any checkpoint
pickled against it - keeps resolving without importing ``sklearn`` at
registration time. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource

if TYPE_CHECKING:
    from drevalpy.components.featurizers.cell_line._proteomics_transformer import (
        ProteomicsMedianCenterAndImputeTransformer,
    )

__all__ = [
    "NormalizedProteomicsCellLineFeaturizer",
    "ProteomicsMedianCenterAndImputeTransformer",
    "log10_and_set_na",
]


def __getattr__(name: str) -> Any:
    """Resolve the re-exported transformer on first access.

    :param name: Attribute being looked up on this module.
    :returns: The requested attribute.
    :raises AttributeError: If *name* is not re-exported here.
    """
    if name == "ProteomicsMedianCenterAndImputeTransformer":
        from drevalpy.components.featurizers.cell_line._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer as _Transformer,
        )

        return _Transformer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def log10_and_set_na(x: np.ndarray) -> np.ndarray:
    """Log10 transform and set NaN for infinite values.

    :param x: input array
    :returns: log10 transformed array with NaN for infinite values
    """
    x = np.log10(x)
    x[np.isinf(x)] = np.nan
    return x


@register(
    "normalizedProteomics",
    description="Proteomics view with log10 transform, median centering, and imputation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class NormalizedProteomicsCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Match sklearn baseline proteomics preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("proteomics",)
    fit_on_unique_ids: ClassVar[bool] = True

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
        from drevalpy.components.featurizers.cell_line._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer,
        )

        super().__init__(view=view)
        self._transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=proteomics_feature_threshold,
            n_features=proteomics_n_features,
            normalization_width=proteomics_normalization_width,
            normalization_downshift=proteomics_normalization_downshift,
        )

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

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the median-centering transformer on log10 training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated cell-line identifiers to fit on.
        :returns: Number of retained proteins.
        """
        self._transformer.fit(log10_and_set_na(self._raw_matrix(source, entity_ids)))
        return len(self._transformer.protein_indices)

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Log10-transform *matrix* and apply the fitted transformer row by row.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Normalized feature matrix.
        """
        _ = source
        rows = [self._transformer.transform(row[None, :])[0] for row in log10_and_set_na(matrix)]
        return np.vstack(rows)

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
        from drevalpy.components.featurizers.cell_line._proteomics_transformer import (
            ProteomicsMedianCenterAndImputeTransformer,
        )

        transformer = state.get("proteomics_transformer")
        if isinstance(transformer, ProteomicsMedianCenterAndImputeTransformer):
            self._transformer = transformer
        self._restore_dense_state(state)
