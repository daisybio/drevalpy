"""Tests for the curation stages around the curve fit: preprocess, normalize, postprocess and the AnnData builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._anndata import _LAYER_METRICS, X_MEASURE, build_anndata
from drevalpy.curation._fit import _build_config, _build_work_items
from drevalpy.curation._normalize import (
    PRE_NORM_SIGNAL_QUALITY,
    _column_names,
    normalize_group,
    restore_signal_quality,
)
from drevalpy.curation._postprocess import _COLUMN_RENAME, DERIVED_METRICS, postprocess
from drevalpy.curation._preprocess import preprocess


def test_returns_groups_of_a_wide_frame_and_group_info(dose_response_df: pd.DataFrame) -> None:
    groups = preprocess(dose_response_df)
    wide_df, group_info = groups[0]
    raw_cols = [c for c in wide_df.columns if str(c).startswith("Raw")]

    assert groups and all(isinstance(item, tuple) and len(item) == 2 for item in groups)
    assert raw_cols
    assert group_info["n_experiments"] == len(raw_cols)
    assert isinstance(group_info["doses"], list)
    assert {"n_replicates"} <= set(group_info)
    assert all(len(name.split("|")) == 2 for name in wide_df["Name"])


def test_replicates_are_counted(dose_response_df_with_replicates: pd.DataFrame) -> None:
    assert preprocess(dose_response_df_with_replicates)[0][1]["n_replicates"] == 2


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess(pd.DataFrame({"drug": ["A"], "sample": ["B"]}))


def test_duplicate_measurements_are_averaged_with_a_warning(dose_response_df: pd.DataFrame) -> None:
    duplicated = pd.concat([dose_response_df, dose_response_df.head(1)], ignore_index=True)

    with pytest.warns(UserWarning, match="Duplicate entries found"):
        groups = preprocess(duplicated)

    assert len(groups[0][0]) == 6


_CONCENTRATIONS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _viability_rows(samples: list[str], drug: str, scale: float) -> list[dict]:
    """Long-form rows for a sigmoid scaled by a per-sample offset (median normalization removes it)."""
    return [
        {
            "drug": drug,
            "sample": sample,
            "concentration": concentration,
            "intensity": scale * (index + 1) * (0.1 + 0.9 / (1 + (concentration / 0.5) ** 1.5)),
        }
        for index, sample in enumerate(samples)
        for concentration in _CONCENTRATIONS
    ]


def build_normalizable_df() -> pd.DataFrame:
    """Twelve curves across two dose-range groups, each on its own intensity scale."""
    rows = _viability_rows([f"CL_{index}" for index in range(6)], "DrugX", scale=1.0)
    rows += _viability_rows([f"CL_{index}" for index in range(6)], "DrugY", scale=3.0)
    frame = pd.DataFrame(rows)
    # Push DrugY into a second dose-range group so the fix is exercised per group.
    frame.loc[frame["drug"] == "DrugY", "concentration"] *= 10.0
    return frame


@pytest.fixture()
def groups() -> list[tuple[pd.DataFrame, dict]]:
    return preprocess(build_normalizable_df())


@pytest.fixture()
def group_and_config(groups: list[tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, dict]:
    df, info = groups[0]
    config = _build_config(info["n_experiments"], info["doses"], info["n_replicates"], normalize=True, fit_speed="fast")
    return df, config


def test_column_names_line_up_with_run_pipeline(group_and_config: tuple[pd.DataFrame, dict]) -> None:
    df, config = group_and_config

    raw, normalized, dosed, controls = _column_names(config)

    assert [name.replace("Raw", "Normalized") for name in raw] == list(normalized)
    assert set(raw) <= set(df.columns)
    assert len(dosed) == len(raw) - 1
    assert list(controls) == ["Raw 0"]


class TestNormalizeGroup:
    def test_raw_columns_are_overwritten_and_helper_columns_and_input_are_left_clean(
        self, group_and_config: tuple[pd.DataFrame, dict]
    ) -> None:
        df, config = group_and_config
        raw, _, _, _ = _column_names(config)
        before = df[raw].to_numpy(copy=True)

        result = normalize_group(df, config)

        assert not np.allclose(result[raw].to_numpy(), before)
        assert not [column for column in result.columns if str(column).startswith("Normalized")]
        np.testing.assert_array_equal(df[raw].to_numpy(), before)
        assert result["Name"].tolist() == df["Name"].tolist()

    def test_pre_normalization_signal_quality_is_carried(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config

        result = normalize_group(df, config)

        np.testing.assert_allclose(result[PRE_NORM_SIGNAL_QUALITY].to_numpy(), np.log2(df["Raw 0"].to_numpy()))

    def test_factors_depend_on_the_rows_but_not_their_order(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config
        raw, _, _, _ = _column_names(config)

        whole = normalize_group(df, config).set_index("Name")[raw]
        shuffled = normalize_group(df.iloc[::-1].reset_index(drop=True), config).set_index("Name")[raw]
        half = normalize_group(df.iloc[:3].reset_index(drop=True), config).set_index("Name")[raw]

        np.testing.assert_allclose(whole.to_numpy(), shuffled.loc[whole.index].to_numpy())
        assert not np.allclose(whole.loc[half.index].to_numpy(), half.to_numpy())


class TestRestoreSignalQuality:
    def test_the_carried_value_replaces_the_post_normalization_one_and_the_carrier_is_dropped(self) -> None:
        fitted = pd.DataFrame({"Signal Quality": [0.0, 0.0], PRE_NORM_SIGNAL_QUALITY: [3.0, 4.0]})

        result = restore_signal_quality(fitted)

        assert result["Signal Quality"].tolist() == [3.0, 4.0]
        assert PRE_NORM_SIGNAL_QUALITY not in result.columns

    def test_a_frame_that_never_was_normalized_passes_through(self) -> None:
        fitted = pd.DataFrame({"Signal Quality": [1.0]})

        assert restore_signal_quality(fitted) is fitted


class TestWorkItemWiring:
    def test_chunks_are_prenormalized_with_the_request_recorded_on_the_group_configs(
        self, groups: list[tuple[pd.DataFrame, dict]]
    ) -> None:
        work_items, configs = _build_work_items(
            groups, max_chunk_size=2, normalize=True, fit_type="OLS", fit_speed="fast"
        )
        plain, _ = _build_work_items(groups, max_chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        assert all(config["Processing"]["normalization"] is False for _, config, _ in work_items)
        assert all(config["Processing"]["normalization"] is True for config in configs)
        assert all(PRE_NORM_SIGNAL_QUALITY in chunk.columns for chunk, _, _ in work_items)
        assert all(PRE_NORM_SIGNAL_QUALITY not in chunk.columns for chunk, _, _ in plain)
        assert [len(chunk) for chunk, _, _ in work_items] == [len(chunk) for chunk, _, _ in plain]


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
