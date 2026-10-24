"""Tests for drevalpy.curation._fit; every test pins ``max_workers=1`` so fitting stays in-process."""

from __future__ import annotations

from typing import NamedTuple

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation import curate
from drevalpy.curation._fit import (
    _build_config,
    _build_work_items,
    _fit_chunk,
    _run_work_items,
    fit_groups,
)
from drevalpy.curation._preprocess import preprocess
from tests.curation.test_stages import build_normalizable_df

_DOSES = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
_Group = list[tuple[pd.DataFrame, dict]]


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
    return _Fitted(groups, fit_groups(groups, max_workers=1, fit_speed="fast"))


@pytest.fixture(scope="session")
def fitted_two_groups() -> _Fitted:
    groups = _build_two_groups()
    return _Fitted(groups, fit_groups(groups, max_workers=1, fit_speed="fast"))


@pytest.fixture()
def recorded_chunk_fits(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Replace ``_fit_chunk`` with a pass-through that records the chunks it saw."""
    seen: list[pd.DataFrame] = []

    def _record(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        seen.append(chunk_df)
        return chunk_df

    monkeypatch.setattr("drevalpy.curation._fit._fit_chunk", _record)
    return seen


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

    fitted = _fit_chunk(wide_df, config)

    assert len(fitted) == len(wide_df)
    assert config["Processing"]["available_max_workers"] == 8


class TestRunWorkItems:
    @pytest.mark.parametrize(("max_chunk_size", "max_workers"), [(2, 1), (10, 4)], ids=["single-core", "single-chunk"])
    def test_stays_in_process(
        self,
        single_group: _Group,
        monkeypatch: pytest.MonkeyPatch,
        recorded_chunk_fits: list[pd.DataFrame],
        max_chunk_size: int,
        max_workers: int,
    ) -> None:
        def _no_pool(*args: object, **kwargs: object) -> None:
            raise AssertionError("ProcessPoolExecutor must not be used on the serial path")

        monkeypatch.setattr("drevalpy.curation._fit.ProcessPoolExecutor", _no_pool)
        work_items, _ = _items(single_group, max_chunk_size)

        assert len(_run_work_items(work_items, max_workers=max_workers)) == len(work_items)

    def test_fits_every_chunk_once_and_preserves_the_group_index(
        self, two_groups: _Group, recorded_chunk_fits: list[pd.DataFrame]
    ) -> None:
        work_items, _ = _items(two_groups, 1)

        results = _run_work_items(work_items, max_workers=1)

        assert [group_idx for _, group_idx in results] == [group_idx for _, _, group_idx in work_items]
        assert [chunk["Name"].tolist() for chunk in recorded_chunk_fits] == [
            chunk["Name"].tolist() for chunk, _, _ in work_items
        ]


@pytest.mark.slow
class TestFitGroups:
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
