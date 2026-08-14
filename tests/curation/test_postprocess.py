"""Tests for drevalpy.curation._postprocess.postprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._postprocess import _COLUMN_RENAME, DERIVED_METRICS, postprocess


def _make_mock_fitted_df() -> pd.DataFrame:
    """Build a frame mimicking curve_curator output for three curves.

    The columns come from :data:`_COLUMN_RENAME` rather than a hand-written list,
    so a metric added there without a value here fails loudly instead of quietly
    exercising a shorter frame.

    :returns: A frame with every column ``postprocess`` renames.
    """
    values: dict[str, list] = {
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
        "pEC50 Error": [0.05, 0.08, 12.0],
        "Curve Slope Error": [0.1, 0.2, 30.0],
        "Curve Front Error": [0.01, 0.02, 3.0],
        "Curve Back Error": [0.01, 0.03, 4.0],
    }
    unset = set(_COLUMN_RENAME) - set(values)
    assert not unset, f"mock frame is missing renamed column(s): {sorted(unset)}"
    return pd.DataFrame(values)


@pytest.fixture()
def fitted_df() -> pd.DataFrame:
    """A frame mimicking curve_curator output for three curves."""
    return _make_mock_fitted_df()


@pytest.fixture()
def metrics(fitted_df: pd.DataFrame) -> pd.DataFrame:
    """``postprocess`` applied to a single group."""
    return postprocess([(fitted_df, {})])


class TestColumns:
    """The shape of the returned flat metrics frame."""

    def test_it_returns_a_dataframe(self, metrics: pd.DataFrame) -> None:
        assert isinstance(metrics, pd.DataFrame)

    @pytest.mark.parametrize("column", ["cell_line", "drug"])
    def test_label_columns_are_present(self, metrics: pd.DataFrame, column: str) -> None:
        assert column in metrics.columns

    @pytest.mark.parametrize("column", list(DERIVED_METRICS))
    def test_derived_metrics_are_present(self, metrics: pd.DataFrame, column: str) -> None:
        assert column in metrics.columns

    @pytest.mark.parametrize("column", sorted(set(_COLUMN_RENAME.values())))
    def test_every_renamed_metric_is_present(self, metrics: pd.DataFrame, column: str) -> None:
        assert column in metrics.columns

    def test_no_raw_curve_curator_names_survive(self, metrics: pd.DataFrame) -> None:
        assert not [column for column in metrics.columns if column.startswith(("Curve ", "Signal "))]

    def test_the_grouping_name_column_is_dropped(self, metrics: pd.DataFrame) -> None:
        assert "Name" not in metrics.columns


class TestPerCurveErrors:
    """The four per-parameter standard errors, previously discarded.

    They are CurveCurator's only per-curve uncertainty estimate, and they were
    silently dropped for as long as ``_COLUMN_RENAME`` did not list them.
    """

    @pytest.mark.parametrize(
        ("source", "renamed"),
        [
            ("pEC50 Error", "pec50_error"),
            ("Curve Slope Error", "slope_error"),
            ("Curve Front Error", "front_error"),
            ("Curve Back Error", "back_error"),
        ],
    )
    def test_each_error_survives_under_its_snake_case_name(
        self, fitted_df: pd.DataFrame, metrics: pd.DataFrame, source: str, renamed: str
    ) -> None:
        np.testing.assert_allclose(metrics[renamed].to_numpy(), fitted_df[source].to_numpy())

    def test_a_missing_error_column_is_an_error_not_a_silent_drop(self, fitted_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="pEC50 Error"):
            postprocess([(fitted_df.drop(columns=["pEC50 Error"]), {})])


class TestDerivedMetrics:
    """``EC50``/``IC50``/``LN_IC50`` are computed here, not read from the fit."""

    def test_ec50_inverts_pec50(self, metrics: pd.DataFrame) -> None:
        np.testing.assert_allclose(metrics.loc[0, "EC50"], 10 ** (-6.0) * 1e6, rtol=1e-5)

    def test_ln_ic50_is_the_log_of_ic50(self, metrics: pd.DataFrame) -> None:
        np.testing.assert_allclose(metrics["LN_IC50"].to_numpy(), np.log(metrics["IC50"].to_numpy()))

    def test_a_curve_with_no_half_maximal_crossing_yields_nan(self, fitted_df: pd.DataFrame) -> None:
        """``CL_A|DrugY`` has ``back = 0.9``, so the curve never reaches 0.5."""
        result = postprocess([(fitted_df, {})])

        assert np.isnan(result.loc[2, "IC50"])


class TestLabels:
    """``Name`` is split back into the labels the caller supplied."""

    def test_the_name_column_is_split_on_the_separator(self, metrics: pd.DataFrame) -> None:
        assert metrics.loc[0, "cell_line"] == "CL_A"
        assert metrics.loc[0, "drug"] == "DrugX"

    def test_native_identifiers_survive_unchanged(self) -> None:
        """The pipeline fits on native IDs and remaps later, so these must pass through."""
        native = _make_mock_fitted_df()
        native["Name"] = ["ACH-000001|DRUG_1047", "SIDM00400|DRUG_1047", "ACH-000001|BRD-K02251932"]

        result = postprocess([(native, {})])

        assert result["cell_line"].tolist() == ["ACH-000001", "SIDM00400", "ACH-000001"]
        assert result["drug"].tolist() == ["DRUG_1047", "DRUG_1047", "BRD-K02251932"]


class TestMultipleGroups:
    """Dose-range groups are concatenated, not merged."""

    def test_rows_from_every_group_are_kept(self, fitted_df: pd.DataFrame) -> None:
        other = fitted_df.copy()
        other["Name"] = ["CL_A|DrugZ", "CL_B|DrugZ", "CL_C|DrugZ"]

        result = postprocess([(fitted_df, {}), (other, {})])

        assert len(result) == 2 * len(fitted_df)

    def test_the_index_is_reset_across_groups(self, fitted_df: pd.DataFrame) -> None:
        result = postprocess([(fitted_df, {}), (fitted_df.copy(), {})])

        assert result.index.tolist() == list(range(len(result)))

    def test_the_input_frame_is_not_mutated(self, fitted_df: pd.DataFrame) -> None:
        before = fitted_df.columns.tolist()

        postprocess([(fitted_df, {})])

        assert fitted_df.columns.tolist() == before
