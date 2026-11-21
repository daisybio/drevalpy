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
    fit_chunk,
    fit_config,
    split,
)
from tests.curation.conftest import fit_table
from tests.curation.test_stages import build_normalizable_df

_DOSES = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
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


def _single_group() -> pd.DataFrame:
    return pd.DataFrame(_curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B", "CL_C"], "DrugX"))


def _two_groups() -> pd.DataFrame:
    """DrugY tops out two decades above DrugX, giving two dose-range groups."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B"], "DrugX")
    rows += _curve_rows([0.001, 0.01, 0.1, 1.0, 100.0], ["CL_A", "CL_B"], "DrugY")
    return pd.DataFrame(rows)


def _viability(n_pairs: int = 6) -> pd.DataFrame:
    """Long viability table: noisy curves for ``n_pairs`` pairs."""
    rng = np.random.default_rng(0)
    rows = []
    for pair in range(n_pairs):
        ec50 = 10 ** rng.uniform(-2, 1)
        for dose in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            response = 1 / (1 + (dose / ec50) ** 1.5) + rng.normal(0, 0.02)
            rows.append({"sample": f"CL{pair}", "drug": f"D{pair % 2}", "concentration": dose, "intensity": response})
    return pd.DataFrame(rows)


class _Fitted(NamedTuple):
    chunks: list[tuple[pd.DataFrame, int]]
    infos: list[dict]
    fitted: list[pd.DataFrame]
    configs: list[dict]
    results: list[pd.DataFrame]


def _fit_groups(viability: pd.DataFrame) -> _Fitted:
    chunks, infos = split(viability, _FAST)
    configs = [fit_config(info, _FAST, normalize=False) for info in infos]
    fitted = [fit_chunk(frame, configs[group]) for frame, group in chunks]
    return _Fitted(chunks, infos, fitted, configs, _assemble_groups(chunks, fitted, configs))


@pytest.fixture(scope="session")
def fitted_single_group() -> _Fitted:
    return _fit_groups(_single_group())


@pytest.fixture(scope="session")
def fitted_two_groups() -> _Fitted:
    return _fit_groups(_two_groups())


class TestFitConfig:
    def test_forwards_fit_type_speed_and_normalize_flag(self) -> None:
        info = {"n_experiments": 6, "doses": _DOSES, "n_replicates": 1}
        options = FitOptions(fit_speed="fast")

        config = fit_config(info, options, normalize=True)

        assert config["Curve Fit"]["type"] == "OLS"
        assert config["Curve Fit"]["speed"] == "fast"
        assert config["Processing"]["normalization"] is True
        assert fit_config(info, options, normalize=False)["Processing"]["normalization"] is False

    @pytest.mark.parametrize(
        ("doses", "expected_max_missing"), [(_DOSES, 1), ([0.0, 0.1, 1.0], 0)], ids=["six-doses", "few-doses"]
    )
    def test_max_missing_scales_with_dose_count(self, doses: list[float], expected_max_missing: int) -> None:
        config = fit_config({"n_experiments": len(doses), "doses": doses, "n_replicates": 1}, _FAST, normalize=False)

        assert config["Processing"]["max_missing"] == expected_max_missing

    def test_experiments_controls_and_curve_curator_defaults(self) -> None:
        config = fit_config({"n_experiments": 12, "doses": _DOSES, "n_replicates": 2}, _FAST, normalize=False)

        assert len(config["Experiment"]["experiments"]) == 12
        assert len(config["Experiment"]["control_experiment"]) == 2
        assert "Dashboard" in config


@pytest.mark.slow
def test_fit_chunk_returns_one_row_per_curve_without_mutating_the_config() -> None:
    chunks, infos = split(_single_group(), _FAST)
    frame, group = chunks[0]
    config = fit_config(infos[group], _FAST, normalize=False)
    config["Processing"]["available_max_workers"] = 8

    fitted = fit_chunk(frame, config)

    assert len(fitted) == len(frame)
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

        assert fit_chunk(pd.DataFrame({"Name": ["a"]}), {"Processing": {}}).empty

    @pytest.mark.parametrize("code", [-1, 0])
    def test_an_exit_code_is_a_failure(self, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
        self._exits_with(monkeypatch, code)

        with pytest.raises(RuntimeError, match=f"exited with code {code}") as info:
            fit_chunk(pd.DataFrame({"Name": ["a"]}), {"Processing": {}})

        assert isinstance(info.value.__cause__, SystemExit)


@pytest.mark.slow
class TestAssembleGroups:
    def test_routes_curves_back_to_their_own_group(self, fitted_two_groups: _Fitted) -> None:
        assert [sorted(df["Name"]) for df in fitted_two_groups.results] == [
            ["CL_A|DrugX", "CL_B|DrugX"],
            ["CL_A|DrugY", "CL_B|DrugY"],
        ]

    def test_keeps_every_curve_and_adds_the_columns_postprocess_consumes(self, fitted_single_group: _Fitted) -> None:
        [fitted_df] = fitted_single_group.results

        assert fitted_df["Name"].tolist() == fitted_single_group.chunks[0][0]["Name"].tolist()
        assert {"pEC50", "Curve Slope", "Curve Front", "Curve Back", "Curve AUC"} <= set(fitted_df.columns)
        assert {"pEC50 Error", "Curve Slope Error", "Curve Front Error", "Curve Back Error"} <= set(fitted_df.columns)
        assert {"Curve Relevance Score", "Curve Regulation"} <= set(fitted_df.columns)

    def test_a_group_whose_every_chunk_is_empty_is_skipped(self, fitted_two_groups: _Fitted) -> None:
        fitted = [fitted_two_groups.fitted[0], pd.DataFrame()]

        assembled = _assemble_groups(fitted_two_groups.chunks, fitted, fitted_two_groups.configs)

        assert [sorted(df["Name"]) for df in assembled] == [["CL_A|DrugX", "CL_B|DrugX"]]

    def test_too_few_curves_for_thresholding_leave_the_sam_columns_nan(
        self, fitted_single_group: _Fitted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def too_few(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise ValueError("Too few curves: less than the number of model parameters")

        monkeypatch.setattr("curve_curator.thresholding.apply_significance_thresholds", too_few)
        fitted = [df.copy() for df in fitted_single_group.fitted]

        [assembled] = _assemble_groups(fitted_single_group.chunks, fitted, fitted_single_group.configs)

        assert assembled["pEC50"].notna().all()
        for column in ("Curve F_Value SAM Corrected", "Curve Relevance Score", "Curve Regulation"):
            assert assembled[column].isna().all()

    def test_any_other_thresholding_error_propagates(
        self, fitted_single_group: _Fitted, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(df: pd.DataFrame, config: dict) -> pd.DataFrame:
            raise ValueError("something else")

        monkeypatch.setattr("curve_curator.thresholding.apply_significance_thresholds", broken)

        with pytest.raises(ValueError, match="something else"):
            _assemble_groups(fitted_single_group.chunks, fitted_single_group.fitted, fitted_single_group.configs)


class TestSplit:
    def test_an_empty_table_has_no_chunks(self) -> None:
        assert split(_viability().iloc[:0], _FAST) == ([], [])

    def test_chunks_partition_the_curves_and_name_their_group(self) -> None:
        chunks, infos = split(_viability(5), FitOptions(fit_speed="fast", max_chunk_size=2))

        assert [len(chunk) for chunk, _ in chunks] == [2, 2, 1]
        assert {group for _, group in chunks} == {0}
        assert infos == [{"n_experiments": 6, "doses": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0], "n_replicates": 1}]

    def test_chunks_restart_their_index(self) -> None:
        chunks, _ = split(_single_group(), FitOptions(fit_speed="fast", max_chunk_size=2))

        assert [chunk.index.tolist() for chunk, _ in chunks] == [[0, 1], [0]]

    def test_each_dose_range_group_has_its_description_and_chunk_index(self) -> None:
        chunks, infos = split(_two_groups(), _FAST)

        assert len(infos) == 2
        assert [group for _, group in chunks] == [0, 1]


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
