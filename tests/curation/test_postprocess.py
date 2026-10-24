"""Tests for drevalpy.curation._postprocess.postprocess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._postprocess import _COLUMN_RENAME, DERIVED_METRICS, postprocess


def _make_mock_fitted_df() -> pd.DataFrame:
    """A frame mimicking curve_curator output; columns come from ``_COLUMN_RENAME`` so a new metric fails loudly."""
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
    return _make_mock_fitted_df()


@pytest.fixture()
def metrics(fitted_df: pd.DataFrame) -> pd.DataFrame:
    return postprocess([(fitted_df, {})])


class TestColumns:
    def test_labels_derived_and_renamed_metrics_are_present_and_raw_names_are_gone(self, metrics: pd.DataFrame) -> None:
        assert {"sample", "drug", *DERIVED_METRICS, *_COLUMN_RENAME.values()} <= set(metrics.columns)
        assert not [column for column in metrics.columns if column.startswith(("Curve ", "Signal "))]
        assert "Name" not in metrics.columns

    @pytest.mark.parametrize(
        ("source", "renamed"),
        [
            ("pEC50 Error", "pec50_error"),
            ("Curve Slope Error", "slope_error"),
            ("Curve Front Error", "front_error"),
            ("Curve Back Error", "back_error"),
        ],
    )
    def test_each_per_curve_error_survives_under_its_snake_case_name(
        self, fitted_df: pd.DataFrame, metrics: pd.DataFrame, source: str, renamed: str
    ) -> None:
        np.testing.assert_allclose(metrics[renamed].to_numpy(), fitted_df[source].to_numpy())

    def test_a_missing_error_column_is_an_error_not_a_silent_drop(self, fitted_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="pEC50 Error"):
            postprocess([(fitted_df.drop(columns=["pEC50 Error"]), {})])


class TestDerivedMetrics:
    def test_ec50_inverts_pec50_and_ln_ic50_is_the_log_of_ic50(self, metrics: pd.DataFrame) -> None:
        np.testing.assert_allclose(metrics.loc[0, "EC50"], 10 ** (-6.0) * 1e6, rtol=1e-5)
        np.testing.assert_allclose(metrics["LN_IC50"].to_numpy(), np.log(metrics["IC50"].to_numpy()))

    def test_a_curve_with_no_half_maximal_crossing_yields_nan(self, metrics: pd.DataFrame) -> None:
        """``CL_A|DrugY`` has ``back = 0.9``, so the curve never reaches 0.5."""
        assert np.isnan(metrics.loc[2, "IC50"])


class TestLabels:
    def test_the_name_column_is_split_on_the_separator(self, metrics: pd.DataFrame) -> None:
        assert (metrics.loc[0, "sample"], metrics.loc[0, "drug"]) == ("CL_A", "DrugX")

    def test_native_identifiers_survive_unchanged(self) -> None:
        native = _make_mock_fitted_df()
        native["Name"] = ["ACH-000001|DRUG_1047", "SIDM00400|DRUG_1047", "ACH-000001|BRD-K02251932"]

        result = postprocess([(native, {})])

        assert result["sample"].tolist() == ["ACH-000001", "SIDM00400", "ACH-000001"]
        assert result["drug"].tolist() == ["DRUG_1047", "DRUG_1047", "BRD-K02251932"]


class TestMultipleGroups:
    def test_groups_are_concatenated_with_a_reset_index_and_the_input_is_not_mutated(
        self, fitted_df: pd.DataFrame
    ) -> None:
        other = fitted_df.copy()
        other["Name"] = ["CL_A|DrugZ", "CL_B|DrugZ", "CL_C|DrugZ"]
        before = fitted_df.columns.tolist()

        result = postprocess([(fitted_df, {}), (other, {})])

        assert len(result) == 2 * len(fitted_df)
        assert result.index.tolist() == list(range(len(result)))
        assert fitted_df.columns.tolist() == before
