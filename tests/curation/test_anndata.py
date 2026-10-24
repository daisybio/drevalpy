"""Tests for drevalpy.curation._anndata.build_anndata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._anndata import _LAYER_METRICS, X_MEASURE, build_anndata


def _make_metrics_df() -> pd.DataFrame:
    """A 2 samples x 2 drugs frame; checked against ``_LAYER_METRICS`` so a new metric fails loudly."""
    values: dict[str, list] = {
        "sample": ["CL_A", "CL_A", "CL_B", "CL_B"],
        "drug": ["DrugX", "DrugY", "DrugX", "DrugY"],
        X_MEASURE: [6.0, 4.0, 5.5, 3.5],
        "EC50": [1.0, 100.0, 3.16, 316.0],
        "IC50": [0.5, 200.0, 1.5, 500.0],
        "LN_IC50": [-0.69, 5.3, 0.4, 6.2],
        "AUC": [0.3, 0.9, 0.4, 0.95],
        "fold_change": [0.9, 0.05, 0.8, 0.03],
        "slope": [1.5, 0.8, 1.2, 0.5],
        "front": [1.0, 1.0, 0.95, 1.0],
        "back": [0.1, 0.9, 0.15, 0.92],
        "R2": [0.99, 0.5, 0.98, 0.4],
        "RMSE": [0.02, 0.05, 0.03, 0.06],
        "p_value": [0.001, 0.5, 0.01, 0.6],
        "log_p_value": [-3.0, -0.3, -2.0, -0.2],
        "f_value": [50.0, 2.0, 30.0, 1.5],
        "f_value_sam": [45.0, 1.5, 28.0, 1.2],
        "relevance_score": [0.95, 0.1, 0.85, 0.05],
        "signal_quality": [0.9, 0.3, 0.85, 0.25],
        "regulation": ["down", "not", "down", "not"],
        "pec50_error": [0.05, 12.0, 0.08, 15.0],
        "slope_error": [0.1, 30.0, 0.2, 40.0],
        "front_error": [0.01, 3.0, 0.02, 4.0],
        "back_error": [0.01, 4.0, 0.03, 5.0],
    }
    unset = set(_LAYER_METRICS) - set(values)
    assert not unset, f"metrics frame is missing layer metric(s): {sorted(unset)}"
    return pd.DataFrame(values)


@pytest.fixture()
def metrics_df() -> pd.DataFrame:
    return _make_metrics_df()


@pytest.fixture()
def adata(metrics_df: pd.DataFrame):
    return build_anndata(metrics_df)


class TestBuildAnndata:
    def test_long_rows_become_a_sorted_samples_by_drugs_matrix(self, adata, metrics_df: pd.DataFrame) -> None:
        assert adata.shape == (2, 2)
        assert list(adata.obs_names) == ["CL_A", "CL_B"]
        assert list(adata.var_names) == ["DrugX", "DrugY"]
        expected = metrics_df.pivot(index="sample", columns="drug", values=X_MEASURE).to_numpy()
        np.testing.assert_allclose(adata.X, expected.astype(np.float32))
        assert adata.X.dtype == np.float32
        assert X_MEASURE not in adata.layers

    def test_an_unmeasured_pair_becomes_nan(self, metrics_df: pd.DataFrame) -> None:
        result = build_anndata(metrics_df.drop(index=1))

        assert np.isnan(np.asarray(result.X)[0, 1])

    def test_every_layer_metric_is_a_float32_layer_and_absent_ones_are_skipped(
        self, adata, metrics_df: pd.DataFrame
    ) -> None:
        assert set(_LAYER_METRICS) <= set(adata.layers)
        assert all(adata.layers[metric].dtype == np.float32 for metric in _LAYER_METRICS)
        assert "AUC" not in build_anndata(metrics_df.drop(columns=["AUC"])).layers

    def test_regulation_is_encoded_numerically_without_mutating_the_input(
        self, adata, metrics_df: pd.DataFrame
    ) -> None:
        assert set(np.unique(adata.layers["regulation"])) == {-1.0, 0.0}
        assert metrics_df["regulation"].tolist() == ["down", "not", "down", "not"]

    @pytest.mark.parametrize("metric", ["pec50_error", "slope_error", "front_error", "back_error"])
    def test_per_curve_error_values_round_trip_through_the_pivot(
        self, metrics_df: pd.DataFrame, adata, metric: str
    ) -> None:
        expected = metrics_df.pivot(index="sample", columns="drug", values=metric).to_numpy()

        np.testing.assert_allclose(adata.layers[metric], expected.astype(np.float32))
        np.testing.assert_array_equal(np.isnan(adata.layers[metric]), np.isnan(adata.X))
