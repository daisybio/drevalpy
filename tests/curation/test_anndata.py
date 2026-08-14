"""Tests for drevalpy.curation._anndata.build_anndata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._anndata import _LAYER_METRICS, X_MEASURE, build_anndata


def _make_metrics_df() -> pd.DataFrame:
    """Build a flat metrics frame covering every layer metric.

    Columns are checked against :data:`_LAYER_METRICS` so a metric added there
    without a value here fails loudly rather than silently skipping its layer.

    :returns: A 2 cell lines x 2 drugs metrics frame.
    """
    values: dict[str, list] = {
        "cell_line": ["CL_A", "CL_A", "CL_B", "CL_B"],
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
    """A flat metrics frame covering every layer metric."""
    return _make_metrics_df()


@pytest.fixture()
def adata(metrics_df: pd.DataFrame):
    """``build_anndata`` applied to that frame."""
    return build_anndata(metrics_df)


class TestShapeAndAxes:
    """Long-form rows become a (cell_lines x drugs) matrix."""

    def test_shape(self, adata) -> None:
        assert adata.shape == (2, 2)

    def test_obs_var_indices_are_sorted_uniques(self, adata) -> None:
        assert list(adata.obs_names) == ["CL_A", "CL_B"]
        assert list(adata.var_names) == ["DrugX", "DrugY"]

    def test_an_unmeasured_pair_becomes_nan(self, metrics_df: pd.DataFrame) -> None:
        result = build_anndata(metrics_df.drop(index=1))

        assert np.isnan(result.X[0, 1])


class TestXMatrix:
    """``X`` holds pEC50; nothing else does."""

    def test_x_is_the_pec50_column(self, metrics_df: pd.DataFrame, adata) -> None:
        expected = metrics_df.pivot(index="cell_line", columns="drug", values=X_MEASURE).to_numpy()

        np.testing.assert_allclose(adata.X, expected.astype(np.float32))

    def test_x_is_float32(self, adata) -> None:
        assert adata.X.dtype == np.float32

    def test_pec50_is_not_duplicated_as_a_layer(self, adata) -> None:
        assert X_MEASURE not in adata.layers


class TestLayers:
    """Every metric present in the frame becomes a float32 layer."""

    @pytest.mark.parametrize("metric", list(_LAYER_METRICS))
    def test_each_layer_metric_is_stored(self, adata, metric: str) -> None:
        assert metric in adata.layers
        assert adata.layers[metric].dtype == np.float32

    def test_a_metric_absent_from_the_frame_is_skipped(self, metrics_df: pd.DataFrame) -> None:
        result = build_anndata(metrics_df.drop(columns=["AUC"]))

        assert "AUC" not in result.layers

    def test_regulation_is_encoded_numerically(self, adata) -> None:
        assert set(np.unique(adata.layers["regulation"])) == {-1.0, 0.0}

    def test_the_input_frame_is_not_mutated(self, metrics_df: pd.DataFrame) -> None:
        build_anndata(metrics_df)

        assert metrics_df["regulation"].tolist() == ["down", "not", "down", "not"]


class TestPerCurveErrorLayers:
    """The recovered per-parameter standard errors reach ``layers`` intact."""

    @pytest.mark.parametrize("metric", ["pec50_error", "slope_error", "front_error", "back_error"])
    def test_values_round_trip_through_the_pivot(self, metrics_df: pd.DataFrame, adata, metric: str) -> None:
        expected = metrics_df.pivot(index="cell_line", columns="drug", values=metric).to_numpy()

        np.testing.assert_allclose(adata.layers[metric], expected.astype(np.float32))

    def test_pec50_error_aligns_with_x(self, adata) -> None:
        """The uncertainty layer must be measured wherever ``X`` is."""
        np.testing.assert_array_equal(np.isnan(adata.layers["pec50_error"]), np.isnan(adata.X))
