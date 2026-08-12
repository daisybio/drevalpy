"""Tests for drevalpy.curation._postprocess.postprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drevalpy.curation._postprocess import postprocess


class TestPostprocess:
    """Tests for drevalpy.curation._postprocess.postprocess."""

    def _make_mock_fitted_df(self) -> pd.DataFrame:
        """Create a DataFrame mimicking curve_curator output."""
        return pd.DataFrame(
            {
                "Name": ["CL_A|DrugX", "CL_B|DrugX", "CL_A|DrugY"],
                "pEC50": [6.0, 5.5, 4.0],
                "Curve Slope": [1.5, 1.2, 0.8],
                "Curve Front": [1.0, 0.95, 1.0],
                "Curve Back": [0.1, 0.15, 0.9],
                "Curve Fold Change": [0.9, 0.8, 0.05],
                "Curve AUC": [0.3, 0.4, 0.9],
                "Curve RMSE": [0.02, 0.03, 0.01],
                "Curve R2": [0.99, 0.98, 0.5],
                "Curve P_Value": [0.001, 0.01, 0.5],
                "Curve Log P_Value": [-3.0, -2.0, -0.3],
                "Curve F_Value": [50.0, 30.0, 2.0],
                "Curve F_Value SAM Corrected": [45.0, 28.0, 1.5],
                "Curve Relevance Score": [0.95, 0.85, 0.1],
                "Curve Regulation": ["down", "down", "not"],
                "Signal Quality": [0.9, 0.85, 0.3],
            }
        )

    def test_returns_dataframe_with_expected_columns(self) -> None:
        mock_df = self._make_mock_fitted_df()
        config = {}
        result = postprocess([(mock_df, config)])

        assert isinstance(result, pd.DataFrame)
        assert "cell_line" in result.columns
        assert "drug" in result.columns
        assert "pEC50" in result.columns
        assert "EC50" in result.columns
        assert "IC50" in result.columns
        assert "LN_IC50" in result.columns

    def test_column_renaming(self) -> None:
        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        assert "slope" in result.columns
        assert "front" in result.columns
        assert "back" in result.columns
        assert "AUC" in result.columns

    def test_ec50_derivation(self) -> None:
        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        expected_ec50 = 10 ** (-6.0) * 1e6
        np.testing.assert_allclose(result.loc[0, "EC50"], expected_ec50, rtol=1e-5)

    def test_cell_line_drug_split(self) -> None:
        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        assert result.loc[0, "cell_line"] == "CL_A"
        assert result.loc[0, "drug"] == "DrugX"
