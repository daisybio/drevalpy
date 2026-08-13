"""Tests for the proteomics median-centering transformer.

The transformer was split out of ``normalized_proteomics`` to keep ``sklearn``
off the ``import drevalpy`` path, so this file also pins the two properties that
split has to preserve: it still satisfies the sklearn estimator protocol, and the
old import path still resolves it.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line._proteomics_transformer import (
    ProteomicsMedianCenterAndImputeTransformer,
)


def _matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )


class TestFit:
    def test_thresholding_keeps_only_sufficiently_complete_features(self) -> None:
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=1)

        transformer.fit(_matrix())

        assert sorted(transformer.protein_indices.tolist()) == [0, 1]

    def test_falls_back_to_the_n_most_complete_features(self) -> None:
        """With ``n_features`` above the complete count, completeness ranking decides."""
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=3)

        transformer.fit(_matrix())

        assert sorted(transformer.protein_indices.tolist()) == [0, 1, 2]

    def test_records_the_mean_of_the_per_row_medians(self) -> None:
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=1)

        transformer.fit(_matrix())

        # Medians over columns 0 and 1: 1.5, 4.5, 7.5.
        np.testing.assert_allclose(transformer.mean_median, 4.5)

    def test_fit_returns_self_for_chaining(self) -> None:
        transformer = ProteomicsMedianCenterAndImputeTransformer()

        assert transformer.fit(_matrix()) is transformer


class TestTransform:
    def test_median_centers_against_the_fitted_mean_median(self) -> None:
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=1)
        transformer.fit(_matrix())

        (row,) = transformer.transform(np.array([[4.0, 5.0, 6.0]]))

        # Row median is 4.5, which already equals mean_median, so values pass through.
        np.testing.assert_allclose(row, [4.0, 5.0])

    def test_imputes_missing_values_from_a_downshifted_normal(self) -> None:
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.0, n_features=3)
        transformer.fit(_matrix())

        (row,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))

        assert not np.isnan(row).any()

    def test_imputation_is_seeded_and_therefore_reproducible(self) -> None:
        """Two identical calls must agree; the seed is per call, not global RNG state."""
        transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.0, n_features=3)
        transformer.fit(_matrix())

        (first,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))
        (second,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))

        np.testing.assert_allclose(first, second)


class TestTheSklearnContract:
    def test_get_params_reports_every_constructor_argument(self) -> None:
        """``BaseEstimator`` is a real base class here, so ``get_params`` must work."""
        params = ProteomicsMedianCenterAndImputeTransformer(n_features=7).get_params()

        assert params["n_features"] == 7
        assert set(params) == {
            "feature_threshold",
            "imputation_seed",
            "n_features",
            "normalization_downshift",
            "normalization_width",
        }

    def test_clone_produces_an_equivalent_unfitted_estimator(self) -> None:
        from sklearn.base import clone

        original = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.5)

        cloned = clone(original)

        assert cloned is not original
        assert cloned.feature_threshold == 0.5
        assert cloned.protein_indices.size == 0


def test_the_featurizer_module_still_re_exports_the_transformer() -> None:
    """The class moved modules; the historical import path is a compatibility promise."""
    from drevalpy.components.featurizers.cell_line import normalized_proteomics

    assert normalized_proteomics.ProteomicsMedianCenterAndImputeTransformer is (
        ProteomicsMedianCenterAndImputeTransformer
    )
