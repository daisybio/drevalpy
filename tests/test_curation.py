"""Tests for the drevalpy.curation submodule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _sigmoid(x: np.ndarray, top: float, bottom: float, ec50: float, slope: float) -> np.ndarray:
    """4-parameter log-logistic sigmoid."""
    return bottom + (top - bottom) / (1 + (x / ec50) ** slope)


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    """Artificial dose-response data: 3 cell lines x 2 drugs."""
    concentrations = [0.001, 0.01, 0.1, 1.0, 10.0]
    cell_lines = ["CL_A", "CL_B", "CL_C"]
    drugs = ["DrugX", "DrugY"]

    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for cl in cell_lines:
        for drug in drugs:
            conc_arr = np.array(concentrations)
            if drug == "DrugX":
                intensity = _sigmoid(conc_arr, top=1.0, bottom=0.1, ec50=0.5, slope=1.5)
            else:
                intensity = np.ones_like(conc_arr) * 0.95

            noise = rng.normal(0, 0.02, size=len(concentrations))
            intensity = np.clip(intensity + noise, 0.01, 1.5)

            for conc, intens in zip(concentrations, intensity, strict=True):
                rows.append({"drug": drug, "cell_line": cl, "concentration": conc, "intensity": intens})

    return pd.DataFrame(rows)


@pytest.fixture()
def dose_response_df_with_replicates(dose_response_df: pd.DataFrame) -> pd.DataFrame:
    """Same data duplicated with replicate column."""
    rng = np.random.default_rng(99)
    rep1 = dose_response_df.copy()
    rep1["replicate"] = 1
    rep2 = dose_response_df.copy()
    rep2["replicate"] = 2
    rep2["intensity"] = rep2["intensity"] + rng.normal(0, 0.01, size=len(rep2))
    return pd.concat([rep1, rep2], ignore_index=True)


class TestPreprocess:
    """Tests for drevalpy.curation._preprocess.preprocess."""

    def test_returns_list_of_tuples(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation._preprocess import preprocess

        result = preprocess(dose_response_df)
        assert isinstance(result, list)
        assert len(result) >= 1
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2

    def test_wide_df_has_expected_columns(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation._preprocess import preprocess

        groups = preprocess(dose_response_df)
        wide_df, _ = groups[0]
        assert "Name" in wide_df.columns
        raw_cols = [c for c in wide_df.columns if str(c).startswith("Raw")]
        assert len(raw_cols) > 0

    def test_group_info_structure(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation._preprocess import preprocess

        groups = preprocess(dose_response_df)
        _, group_info = groups[0]
        assert "n_experiments" in group_info
        assert "doses" in group_info
        assert "n_replicates" in group_info
        assert isinstance(group_info["doses"], list)
        assert group_info["n_experiments"] == len([c for c in groups[0][0].columns if str(c).startswith("Raw")])

    def test_missing_columns_raises(self) -> None:
        from drevalpy.curation._preprocess import preprocess

        bad_df = pd.DataFrame({"drug": ["A"], "cell_line": ["B"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(bad_df)

    def test_with_replicates(self, dose_response_df_with_replicates: pd.DataFrame) -> None:
        from drevalpy.curation._preprocess import preprocess

        groups = preprocess(dose_response_df_with_replicates)
        _, group_info = groups[0]
        assert group_info["n_replicates"] == 2

    def test_name_column_format(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation._preprocess import preprocess

        groups = preprocess(dose_response_df)
        wide_df, _ = groups[0]
        for name in wide_df["Name"]:
            parts = name.split("|")
            assert len(parts) == 2


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
        from drevalpy.curation._postprocess import postprocess

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
        from drevalpy.curation._postprocess import postprocess

        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        assert "slope" in result.columns
        assert "front" in result.columns
        assert "back" in result.columns
        assert "AUC" in result.columns

    def test_ec50_derivation(self) -> None:
        from drevalpy.curation._postprocess import postprocess

        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        expected_ec50 = 10 ** (-6.0) * 1e6
        np.testing.assert_allclose(result.loc[0, "EC50"], expected_ec50, rtol=1e-5)

    def test_cell_line_drug_split(self) -> None:
        from drevalpy.curation._postprocess import postprocess

        mock_df = self._make_mock_fitted_df()
        result = postprocess([(mock_df, {})])

        assert result.loc[0, "cell_line"] == "CL_A"
        assert result.loc[0, "drug"] == "DrugX"


class TestBuildAnndata:
    """Tests for drevalpy.curation._anndata.build_anndata."""

    def _make_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_line": ["CL_A", "CL_A", "CL_B", "CL_B"],
                "drug": ["DrugX", "DrugY", "DrugX", "DrugY"],
                "pEC50": [6.0, 4.0, 5.5, 3.5],
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
            }
        )

    def test_shape(self) -> None:
        from drevalpy.curation._anndata import build_anndata

        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert adata.shape == (2, 2)  # 2 cell lines x 2 drugs

    def test_x_is_pec50(self) -> None:
        from drevalpy.curation._anndata import build_anndata

        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert adata.X.dtype == np.float32
        assert not np.all(np.isnan(adata.X))

    def test_layers_present(self) -> None:
        from drevalpy.curation._anndata import build_anndata

        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert "EC50" in adata.layers
        assert "IC50" in adata.layers
        assert "AUC" in adata.layers
        assert "slope" in adata.layers
        assert "regulation" in adata.layers

    def test_obs_var_indices(self) -> None:
        from drevalpy.curation._anndata import build_anndata

        df = self._make_metrics_df()
        adata = build_anndata(df)

        assert list(adata.obs_names) == ["CL_A", "CL_B"]
        assert list(adata.var_names) == ["DrugX", "DrugY"]

    def test_regulation_encoded(self) -> None:
        from drevalpy.curation._anndata import build_anndata

        df = self._make_metrics_df()
        adata = build_anndata(df)

        reg_layer = adata.layers["regulation"]
        assert -1.0 in reg_layer
        assert 0.0 in reg_layer


class TestEndToEnd:
    """Integration test for drevalpy.curation.curate."""

    def test_curate_returns_anndata(self, dose_response_df: pd.DataFrame) -> None:
        import anndata

        from drevalpy.curation import curate

        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert isinstance(adata, anndata.AnnData)
        assert adata.shape == (3, 2)

    def test_curate_x_not_all_nan(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation import curate

        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert not np.all(np.isnan(adata.X))

    def test_curate_layers_exist(self, dose_response_df: pd.DataFrame) -> None:
        from drevalpy.curation import curate

        adata = curate(dose_response_df, cores=1, fit_speed="fast")

        assert len(adata.layers) > 0
        assert "EC50" in adata.layers
        assert "AUC" in adata.layers


class TestCLI:
    """Test the CLI curate command via typer's CliRunner."""

    def test_curate_csv_to_h5ad(self, dose_response_df: pd.DataFrame, tmp_path) -> None:
        from typer.testing import CliRunner

        from drevalpy.cli.main import app

        input_csv = tmp_path / "input.csv"
        output_h5ad = tmp_path / "output.h5ad"
        dose_response_df.to_csv(input_csv, index=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["curate", str(input_csv), str(output_h5ad), "--cores", "1", "--fit-speed", "fast"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_h5ad.exists()
