"""Tests for drevalpy.curation._fit; every test fits in this process."""

from __future__ import annotations

from typing import NamedTuple

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation import curate
from drevalpy.curation._fit import (
    STATUS_COLUMN,
    FitOptions,
    Status,
    _assemble_groups,
    _build_config,
    _build_work_items,
    chunk_config,
    fit_chunk,
    fit_table,
    group_config,
    split,
)
from drevalpy.curation._preprocess import preprocess
from tests.curation.test_stages import build_normalizable_df

_DOSES = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
_Group = list[tuple[pd.DataFrame, dict]]
_FAST = FitOptions(fit_speed="fast")


def _curve_rows(concentrations: list[float], samples: list[str], drug: str) -> list[dict]:
    return [
        {
            "drug": drug,
            "sample": sample,
            "concentration": concentration,
            "intensity": 0.1 + 0.9 / (1 + (concentration / 0.5) ** 1.5),
        }
        for sample in samples
        for concentration in concentrations
    ]


def _build_single_group() -> _Group:
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B", "CL_C"], "DrugX")
    return preprocess(pd.DataFrame(rows))


def _build_two_groups() -> _Group:
    """DrugY tops out two decades above DrugX, giving two dose-range groups."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B"], "DrugX")
    rows += _curve_rows([0.001, 0.01, 0.1, 1.0, 100.0], ["CL_A", "CL_B"], "DrugY")
    return preprocess(pd.DataFrame(rows))


def _viability(n_pairs: int = 6) -> pd.DataFrame:
    """Long viability table: noisy curves for ``n_pairs`` pairs, two measurements per dose."""
    rng = np.random.default_rng(0)
    rows = []
    for pair in range(n_pairs):
        ec50 = 10 ** rng.uniform(-2, 1)
        for dose in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            for _ in range(2):
                response = 1 / (1 + (dose / ec50) ** 1.5) + rng.normal(0, 0.02)
                rows.append(
                    {"sample": f"CL{pair}", "drug": f"D{pair % 2}", "concentration": dose, "intensity": response}
                )
    return pd.DataFrame(rows)


def _fit_groups(groups: _Group) -> _Group:
    work_items, configs = _items(groups, 2000)
    return _assemble_groups(work_items, [fit_chunk((chunk, config)) for chunk, config, _ in work_items], configs)


class _Fitted(NamedTuple):
    groups: _Group
    results: _Group


@pytest.fixture()
def single_group() -> _Group:
    return _build_single_group()


@pytest.fixture()
def two_groups() -> _Group:
    return _build_two_groups()


@pytest.fixture(scope="session")
def fitted_single_group() -> _Fitted:
    groups = _build_single_group()
    return _Fitted(groups, _fit_groups(groups))


@pytest.fixture(scope="session")
def fitted_two_groups() -> _Fitted:
    groups = _build_two_groups()
    return _Fitted(groups, _fit_groups(groups))


def _items(groups: _Group, max_chunk_size: int):
    return _build_work_items(groups, max_chunk_size=max_chunk_size, normalize=False, fit_type="OLS", fit_speed="fast")


class TestBuildConfig:
    def test_forwards_fit_type_speed_and_normalize_flag(self) -> None:
        # "MLE" is rejected by the public curate(), but the private plumbing still forwards it.
        config = _build_config(6, _DOSES, 1, fit_type="MLE", fit_speed="fast", normalize=True)

        assert config["Curve Fit"]["type"] == "MLE"
        assert config["Curve Fit"]["speed"] == "fast"
        assert config["Processing"]["normalization"] is True

    @pytest.mark.parametrize(
        ("doses", "expected_max_missing"), [(_DOSES, 1), ([0.0, 0.1, 1.0], 0)], ids=["six-doses", "few-doses"]
    )
    def test_max_missing_scales_with_dose_count(self, doses: list[float], expected_max_missing: int) -> None:
        config = _build_config(len(doses), doses, 1)

        assert config["Processing"]["max_missing"] == expected_max_missing

    def test_experiments_controls_and_curve_curator_defaults(self) -> None:
        config = _build_config(12, _DOSES, 2)

        assert len(config["Experiment"]["experiments"]) == 12
        assert len(config["Experiment"]["control_experiment"]) == 2
        assert "Dashboard" in config


class TestBuildWorkItems:
    def test_chunks_partition_the_group_and_restart_their_index(self, single_group: _Group) -> None:
        work_items, _ = _items(single_group, 2)

        assert len(work_items) == 2
        assert sum(len(chunk) for chunk, _, _ in work_items) == len(single_group[0][0])
        assert work_items[1][0].index.tolist() == [0]

    def test_one_config_per_group_and_each_chunk_carries_its_group_index(self, two_groups: _Group) -> None:
        work_items, configs = _items(two_groups, 10)

        assert len(configs) == 2
        assert [group_idx for _, _, group_idx in work_items] == [0, 1]


@pytest.mark.slow
def test_fit_chunk_returns_one_row_per_curve_without_mutating_the_config(single_group: _Group) -> None:
    wide_df, group_info = single_group[0]
    config = _build_config(group_info["n_experiments"], group_info["doses"], 1, fit_speed="fast")
    config["Processing"]["available_max_workers"] = 8

    fitted = fit_chunk((wide_df, config))

    assert len(fitted) == len(wide_df)
    assert config["Processing"]["available_max_workers"] == 8


class TestFitChunkExits:
    @staticmethod
    def _exits_with(monkeypatch: pytest.MonkeyPatch, code: object) -> None:
        def run_pipeline(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise SystemExit(code)

        monkeypatch.setattr("curve_curator.quantification.run_pipeline", run_pipeline)

    def test_a_bare_exit_means_no_curve_is_left_and_gives_an_empty_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._exits_with(monkeypatch, None)

        assert fit_chunk((pd.DataFrame({"Name": ["a"]}), {"Processing": {}})).empty

    @pytest.mark.parametrize("code", [-1, 0])
    def test_an_exit_code_is_a_failure(self, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
        self._exits_with(monkeypatch, code)

        with pytest.raises(RuntimeError, match=f"exited with code {code}") as info:
            fit_chunk((pd.DataFrame({"Name": ["a"]}), {"Processing": {}}))

        assert isinstance(info.value.__cause__, SystemExit)


@pytest.mark.slow
class TestAssembleGroups:
    def test_routes_curves_back_to_their_own_group(self, fitted_two_groups: _Fitted) -> None:
        assert [sorted(df["Name"]) for df, _ in fitted_two_groups.results] == [
            ["CL_A|DrugX", "CL_B|DrugX"],
            ["CL_A|DrugY", "CL_B|DrugY"],
        ]

    def test_keeps_every_curve_and_adds_the_columns_postprocess_consumes(self, fitted_single_group: _Fitted) -> None:
        fitted_df, config = fitted_single_group.results[0]

        assert fitted_df["Name"].tolist() == fitted_single_group.groups[0][0]["Name"].tolist()
        assert {"pEC50", "Curve Slope", "Curve Front", "Curve Back", "Curve AUC"} <= set(fitted_df.columns)
        assert {"pEC50 Error", "Curve Slope Error", "Curve Front Error", "Curve Back Error"} <= set(fitted_df.columns)
        assert {"Curve Relevance Score", "Curve Regulation"} <= set(fitted_df.columns)
        assert config["Curve Fit"]["speed"] == "fast"
        assert config["Experiment"]["doses"].tolist() == fitted_single_group.groups[0][1]["doses"]

    def test_a_group_whose_every_chunk_is_empty_is_skipped(self, two_groups: _Group) -> None:
        work_items, configs = _items(two_groups, 2000)
        fitted = [fit_chunk((work_items[0][0], work_items[0][1])), pd.DataFrame()]

        assembled = _assemble_groups(work_items, fitted, configs)

        assert [sorted(df["Name"]) for df, _ in assembled] == [["CL_A|DrugX", "CL_B|DrugX"]]

    def test_too_few_curves_for_thresholding_leave_the_sam_columns_nan(
        self, single_group: _Group, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def too_few(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise ValueError("Too few curves: less than the number of model parameters")

        work_items, configs = _items(single_group, 2000)
        fitted = [fit_chunk((chunk, config)) for chunk, config, _ in work_items]
        monkeypatch.setattr("curve_curator.thresholding.apply_significance_thresholds", too_few)

        [(assembled, _)] = _assemble_groups(work_items, fitted, configs)

        assert assembled["pEC50"].notna().all()
        for column in ("Curve F_Value SAM Corrected", "Curve Relevance Score", "Curve Regulation"):
            assert assembled[column].isna().all()

    def test_any_other_thresholding_error_propagates(
        self, single_group: _Group, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise ValueError("something else")

        work_items, configs = _items(single_group, 2000)
        fitted = [fit_chunk((chunk, config)) for chunk, config, _ in work_items]
        monkeypatch.setattr("curve_curator.thresholding.apply_significance_thresholds", broken)

        with pytest.raises(ValueError, match="something else"):
            _assemble_groups(work_items, fitted, configs)


class TestSplit:
    def test_an_empty_table_has_no_chunks(self) -> None:
        assert split(_viability().iloc[:0], _FAST) == ([], [])

    def test_chunks_partition_the_curves_and_name_their_group(self) -> None:
        chunks, infos = split(_viability(5), FitOptions(fit_speed="fast", max_chunk_size=2))

        assert [len(chunk) for chunk, _ in chunks] == [2, 2, 1]
        assert {group for _, group in chunks} == {0}
        assert infos == [{"n_experiments": 6, "doses": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0], "n_replicates": 1}]

    def test_a_chunk_never_normalizes_but_its_group_does(self) -> None:
        options = FitOptions(normalize=True)
        info = {"n_experiments": 6, "doses": _DOSES, "n_replicates": 1}

        assert chunk_config(info, options)["Processing"]["normalization"] is False
        assert group_config(info, options)["Processing"]["normalization"] is True


@pytest.mark.slow
def test_the_result_does_not_depend_on_the_chunk_size() -> None:
    one = fit_table(_viability(), FitOptions(fit_speed="fast", max_chunk_size=2000))
    many = fit_table(_viability(), FitOptions(fit_speed="fast", max_chunk_size=2))

    pd.testing.assert_frame_equal(one, many)


@pytest.mark.slow
class TestFitTableStatus:
    def test_every_attempted_curve_has_a_fitted_row(self) -> None:
        fits = fit_table(_viability(), _FAST)

        assert set(fits[STATUS_COLUMN]) == {Status.FITTED}
        assert {"pEC50", "AUC", "signal_quality"} <= set(fits.columns)

    def test_a_curve_curvecurator_drops_comes_back_as_failed_with_nan_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from curve_curator import quantification

        real = quantification.run_pipeline

        def drops_cl0(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
            result = real(chunk_df, config)
            return result[~result["Name"].str.startswith("CL0|")]

        monkeypatch.setattr("curve_curator.quantification.run_pipeline", drops_cl0)
        fits = fit_table(_viability(), _FAST)

        failed = fits[fits[STATUS_COLUMN] == Status.FIT_FAILED]
        assert failed["sample"].tolist() == ["CL0"]
        assert failed["pEC50"].isna().all()

    def test_a_chunk_where_every_curve_is_dropped_gives_failed_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_curve_left(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise SystemExit

        monkeypatch.setattr("curve_curator.quantification.run_pipeline", no_curve_left)
        fits = fit_table(_viability(2), _FAST)

        assert fits[["sample", "drug"]].values.tolist() == [["CL0", "D0"], ["CL1", "D1"]]
        assert set(fits[STATUS_COLUMN]) == {Status.FIT_FAILED}

    def test_an_empty_table_gives_an_empty_result(self) -> None:
        fits = fit_table(_viability().iloc[:0], _FAST)

        assert fits.empty
        assert list(fits.columns) == ["sample", "drug", STATUS_COLUMN]


@pytest.mark.slow
class TestNormalizedFitIsCoreCountIndependent:
    """Regression: ``normalize=True`` used to run per parallel chunk, so output depended on the worker count."""

    @staticmethod
    def _curate(*, max_workers: int, normalize: bool) -> anndata.AnnData:
        return curate(build_normalizable_df(), max_workers=max_workers, normalize=normalize, fit_speed="fast")

    def test_serial_and_pooled_fits_are_identical(self) -> None:
        serial = self._curate(max_workers=1, normalize=True)
        pooled = self._curate(max_workers=4, normalize=True)

        assert serial.obs_names.tolist() == pooled.obs_names.tolist()
        assert serial.var_names.tolist() == pooled.var_names.tolist()
        np.testing.assert_array_equal(serial.X, pooled.X)
        assert set(serial.layers) == set(pooled.layers)
        for name in serial.layers.keys():
            np.testing.assert_array_equal(serial.layers[name], pooled.layers[name], err_msg=name)

    def test_normalization_changes_the_result_but_signal_quality_reflects_the_raw_controls(self) -> None:
        normalized = self._curate(max_workers=1, normalize=True)
        plain = self._curate(max_workers=1, normalize=False)

        assert not np.allclose(normalized.X, plain.X, equal_nan=True)
        np.testing.assert_allclose(normalized.layers["signal_quality"], plain.layers["signal_quality"])
