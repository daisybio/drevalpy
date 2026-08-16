"""Tests for drevalpy.curation._fit.

Every test pins ``max_workers=1`` (or a single work item) so the fitting stays in the
calling process: the parallel path spawns a ``ProcessPoolExecutor``, which would
re-import curve_curator per worker for no additional coverage.

A real CurveCurator fit costs ~0.5s, so the tests that only *read* a fitted frame
share one session-scoped fit (:class:`_Fitted` below) instead of repeating it.
Two kinds of test deliberately keep their own fit and are marked as such in place:
those asserting a mutation or lifecycle side effect of the call, and those where
the call itself - not its result - is the subject.
"""

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
from tests.curation.test_normalize import build_normalizable_df

_DOSES = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


def _curve_rows(concentrations: list[float], cell_lines: list[str], drug: str) -> list[dict]:
    """Build long-form rows following a well-behaved sigmoid, one row per dose."""
    return [
        {
            "drug": drug,
            "cell_line": cell_line,
            "concentration": concentration,
            "intensity": 0.1 + 0.9 / (1 + (concentration / 0.5) ** 1.5),
        }
        for cell_line in cell_lines
        for concentration in concentrations
    ]


def _build_single_group() -> list[tuple[pd.DataFrame, dict]]:
    """One dose-range group holding three curves."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B", "CL_C"], "DrugX")
    return preprocess(pd.DataFrame(rows))


def _build_two_groups() -> list[tuple[pd.DataFrame, dict]]:
    """Two dose-range groups: DrugY tops out two decades above DrugX."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B"], "DrugX")
    rows += _curve_rows([0.001, 0.01, 0.1, 1.0, 100.0], ["CL_A", "CL_B"], "DrugY")
    return preprocess(pd.DataFrame(rows))


class _Fitted(NamedTuple):
    """A completed ``fit_groups`` run together with the groups it was given.

    Both halves come from the same session-scoped build, so a test may compare the
    results against ``groups`` without rebuilding either.
    """

    groups: list[tuple[pd.DataFrame, dict]]
    results: list[tuple[pd.DataFrame, dict]]


@pytest.fixture()
def single_group() -> list[tuple[pd.DataFrame, dict]]:
    """One dose-range group holding three curves, private to the requesting test."""
    return _build_single_group()


@pytest.fixture()
def two_groups() -> list[tuple[pd.DataFrame, dict]]:
    """Two dose-range groups, private to the requesting test."""
    return _build_two_groups()


@pytest.fixture(scope="session")
def fitted_single_group() -> _Fitted:
    """``fit_groups`` over one group, fitted once and shared read-only."""
    groups = _build_single_group()
    return _Fitted(groups, fit_groups(groups, max_workers=1, fit_speed="fast"))


@pytest.fixture(scope="session")
def fitted_two_groups() -> _Fitted:
    """``fit_groups`` over two dose-range groups, fitted once and shared read-only."""
    groups = _build_two_groups()
    return _Fitted(groups, fit_groups(groups, max_workers=1, fit_speed="fast"))


@pytest.fixture(scope="session")
def fitted_chunk() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ``_fit_chunk`` call, as ``(input wide_df, fitted frame)``."""
    wide_df, group_info = _build_single_group()[0]
    config = _build_config(group_info["n_experiments"], group_info["doses"], 1, fit_speed="fast")
    return wide_df, _fit_chunk(wide_df, config)


@pytest.fixture()
def recorded_chunk_fits(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    """Replace ``_fit_chunk`` with a pass-through that records the chunks it saw.

    ``_run_work_items`` only dispatches and re-labels; substituting the fit keeps
    the real dispatch under test - the returned results still come from this stub,
    so a missed work item still shows up as a length mismatch - while dropping a
    CurveCurator fit per chunk that no assertion here inspects.
    """
    seen: list[pd.DataFrame] = []

    def _record(chunk_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        seen.append(chunk_df)
        return chunk_df

    monkeypatch.setattr("drevalpy.curation._fit._fit_chunk", _record)
    return seen


class TestBuildConfig:
    """The curve_curator config assembled from a preprocess group_info."""

    def test_forwards_fit_type_and_speed(self) -> None:
        # "MLE" is rejected by the public curate() entry point, but the private
        # plumbing still forwards it so re-enabling is a one-line change once the
        # curve_curator fork accepts the 'weights' argument again.
        config = _build_config(6, _DOSES, 1, fit_type="MLE", fit_speed="fast")

        assert config["Curve Fit"]["type"] == "MLE"
        assert config["Curve Fit"]["speed"] == "fast"

    def test_forwards_normalize_flag(self) -> None:
        config = _build_config(6, _DOSES, 1, normalize=True)

        assert config["Processing"]["normalization"] is True

    @pytest.mark.parametrize(
        ("doses", "expected_max_missing"),
        [
            pytest.param(_DOSES, 1, id="six-doses-tolerates-one"),
            pytest.param([0.0, 0.1, 1.0], 0, id="fewer-than-five-doses-tolerates-none"),
        ],
    )
    def test_max_missing_scales_with_dose_count(self, doses: list[float], expected_max_missing: int) -> None:
        config = _build_config(len(doses), doses, 1)

        assert config["Processing"]["max_missing"] == expected_max_missing

    def test_experiment_indices_cover_every_column(self) -> None:
        config = _build_config(6, _DOSES, 1)

        assert len(config["Experiment"]["experiments"]) == 6

    def test_control_experiments_follow_replicate_count(self) -> None:
        config = _build_config(12, _DOSES, 2)

        assert len(config["Experiment"]["control_experiment"]) == 2

    def test_defaults_are_filled_in_by_curve_curator(self) -> None:
        """``set_default_values`` adds sections the caller never writes."""
        config = _build_config(6, _DOSES, 1)

        assert "Dashboard" in config


class TestBuildWorkItems:
    """Chunking of groups into independently fittable units."""

    def test_splits_a_group_into_ceil_chunks(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(
            single_group, max_chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        assert len(work_items) == 2

    def test_chunks_partition_the_group(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(
            single_group, max_chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        assert sum(len(chunk) for chunk, _, _ in work_items) == len(single_group[0][0])

    def test_chunk_index_is_reset(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(
            single_group, max_chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        _, trailing_chunk = work_items[0][0], work_items[1][0]
        assert trailing_chunk.index.tolist() == [0]

    def test_one_config_per_group(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        _, configs = _build_work_items(two_groups, max_chunk_size=10, normalize=False, fit_type="OLS", fit_speed="fast")

        assert len(configs) == 2

    def test_each_chunk_carries_its_group_index(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(
            two_groups, max_chunk_size=10, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        assert [group_idx for _, _, group_idx in work_items] == [0, 1]


class TestFitChunk:
    """Single-chunk fitting.

    Extended tier: both tests run a real CurveCurator fit, ~1s together.
    """

    pytestmark = pytest.mark.slow

    def test_does_not_mutate_the_shared_config(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        # Keeps its own fit on purpose: the assertion is about what the call did to
        # the config it was handed, which a shared fitted result cannot show.
        wide_df, group_info = single_group[0]
        config = _build_config(group_info["n_experiments"], group_info["doses"], 1, fit_speed="fast")
        config["Processing"]["available_max_workers"] = 8

        _fit_chunk(wide_df, config)

        assert config["Processing"]["available_max_workers"] == 8

    def test_returns_one_row_per_curve(self, fitted_chunk: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        wide_df, fitted = fitted_chunk

        assert len(fitted) == len(wide_df)


class TestRunWorkItems:
    """Dispatch between the serial and the pooled fitting paths."""

    @pytest.mark.parametrize(
        ("max_chunk_size", "max_workers"),
        [
            pytest.param(2, 1, id="single-core-many-chunks"),
            pytest.param(10, 4, id="many-cores-single-chunk"),
        ],
    )
    def test_stays_in_process(
        self,
        single_group: list[tuple[pd.DataFrame, dict]],
        monkeypatch: pytest.MonkeyPatch,
        recorded_chunk_fits: list[pd.DataFrame],
        max_chunk_size: int,
        max_workers: int,
    ) -> None:
        def _no_pool(*args: object, **kwargs: object) -> None:
            raise AssertionError("ProcessPoolExecutor must not be used on the serial path")

        monkeypatch.setattr("drevalpy.curation._fit.ProcessPoolExecutor", _no_pool)
        work_items, _ = _build_work_items(
            single_group, max_chunk_size=max_chunk_size, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        results = _run_work_items(work_items, max_workers=max_workers)

        assert len(results) == len(work_items)

    def test_preserves_group_index_per_chunk(
        self, two_groups: list[tuple[pd.DataFrame, dict]], recorded_chunk_fits: list[pd.DataFrame]
    ) -> None:
        work_items, _ = _build_work_items(
            two_groups, max_chunk_size=1, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        results = _run_work_items(work_items, max_workers=1)

        assert [group_idx for _, group_idx in results] == [group_idx for _, _, group_idx in work_items]

    def test_fits_every_chunk_exactly_once(
        self, two_groups: list[tuple[pd.DataFrame, dict]], recorded_chunk_fits: list[pd.DataFrame]
    ) -> None:
        """Guards the stub above: a dropped work item must not look like a pass."""
        work_items, _ = _build_work_items(
            two_groups, max_chunk_size=1, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        _run_work_items(work_items, max_workers=1)

        assert [chunk["Name"].tolist() for chunk in recorded_chunk_fits] == [
            chunk["Name"].tolist() for chunk, _, _ in work_items
        ]


class TestFitGroups:
    """The public entry point.

    Extended tier: the session-scoped ``fitted_*`` fixtures behind these are real
    CurveCurator fits (~1.5s). Because the fits are shared, the cost only goes away
    when the whole class is deselected - hence a class-level marker.
    """

    pytestmark = pytest.mark.slow

    def test_returns_one_result_per_group(self, fitted_two_groups: _Fitted) -> None:
        assert len(fitted_two_groups.results) == 2

    def test_keeps_every_curve(self, fitted_single_group: _Fitted) -> None:
        fitted_df, _ = fitted_single_group.results[0]

        assert fitted_df["Name"].tolist() == fitted_single_group.groups[0][0]["Name"].tolist()

    def test_routes_curves_back_to_their_own_group(self, fitted_two_groups: _Fitted) -> None:
        assert [sorted(df["Name"]) for df, _ in fitted_two_groups.results] == [
            ["CL_A|DrugX", "CL_B|DrugX"],
            ["CL_A|DrugY", "CL_B|DrugY"],
        ]

    def test_adds_the_curve_parameters_postprocess_consumes(self, fitted_single_group: _Fitted) -> None:
        fitted_df, _ = fitted_single_group.results[0]

        assert {"pEC50", "Curve Slope", "Curve Front", "Curve Back", "Curve AUC"} <= set(fitted_df.columns)

    def test_emits_the_per_curve_parameter_errors(self, fitted_single_group: _Fitted) -> None:
        """CurveCurator computes these on every fit; ``_postprocess`` keeps them."""
        fitted_df, _ = fitted_single_group.results[0]

        assert {"pEC50 Error", "Curve Slope Error", "Curve Front Error", "Curve Back Error"} <= set(fitted_df.columns)

    def test_applies_significance_thresholds(self, fitted_single_group: _Fitted) -> None:
        """Only ``thresholding.apply_significance_thresholds`` adds these columns."""
        fitted_df, _ = fitted_single_group.results[0]

        assert {"Curve Relevance Score", "Curve Regulation"} <= set(fitted_df.columns)

    def test_returns_the_config_each_group_was_fitted_with(self, fitted_single_group: _Fitted) -> None:
        _, config = fitted_single_group.results[0]

        assert config["Curve Fit"]["speed"] == "fast"
        assert config["Experiment"]["doses"].tolist() == fitted_single_group.groups[0][1]["doses"]


class TestNormalizedFitIsCoreCountIndependent:
    """The regression guard for the per-chunk normalization bug.

    ``normalize=True`` used to run inside every parallel chunk, so a dataset got
    one set of median-derived normalization factors per chunk and its output
    depended on the worker count. ``max_workers=1`` and ``max_workers=4`` chunk
    the same group into one and four pieces respectively, so agreeing here is
    exactly the property that used to fail.

    Driven through :func:`drevalpy.curation.curate` because that is the only
    public entry point; the AnnData it returns is indexed by cell line and drug,
    so ``X`` and the layers line up pair-for-pair without any sorting.

    Extended tier: two real multi-curve CurveCurator fits, one of them across a
    process pool.
    """

    pytestmark = pytest.mark.slow

    @staticmethod
    def _curate(*, max_workers: int, normalize: bool) -> anndata.AnnData:
        """Curate the normalizable dataset at a given core count."""
        return curate(build_normalizable_df(), max_workers=max_workers, normalize=normalize, fit_speed="fast")

    @pytest.fixture(scope="class")
    def normalized_fits(self) -> tuple[anndata.AnnData, anndata.AnnData]:
        """The same normalized dataset curated at ``max_workers=1`` and ``max_workers=4``."""
        return self._curate(max_workers=1, normalize=True), self._curate(max_workers=4, normalize=True)

    def test_the_same_curves_come_back(self, normalized_fits: tuple[anndata.AnnData, anndata.AnnData]) -> None:
        serial, pooled = normalized_fits

        assert serial.obs_names.tolist() == pooled.obs_names.tolist()
        assert serial.var_names.tolist() == pooled.var_names.tolist()

    def test_every_metric_is_identical(self, normalized_fits: tuple[anndata.AnnData, anndata.AnnData]) -> None:
        serial, pooled = normalized_fits

        np.testing.assert_array_equal(serial.X, pooled.X)
        assert set(serial.layers) == set(pooled.layers)
        for name in serial.layers:
            np.testing.assert_array_equal(serial.layers[name], pooled.layers[name], err_msg=name)

    def test_normalization_actually_changed_the_result(self) -> None:
        """Otherwise the equality above would hold for a no-op implementation."""
        normalized = self._curate(max_workers=1, normalize=True)
        plain = self._curate(max_workers=1, normalize=False)

        assert not np.allclose(normalized.X, plain.X, equal_nan=True)

    def test_signal_quality_reflects_the_raw_controls(self) -> None:
        """Normalization overwrites the raw columns, so this is restored explicitly."""
        normalized = self._curate(max_workers=1, normalize=True)
        plain = self._curate(max_workers=1, normalize=False)

        np.testing.assert_allclose(normalized.layers["signal_quality"], plain.layers["signal_quality"])
