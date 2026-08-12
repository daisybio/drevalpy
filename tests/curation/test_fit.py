"""Tests for drevalpy.curation._fit.

Every test pins ``cores=1`` (or a single work item) so the fitting stays in the
calling process: the parallel path spawns a ``ProcessPoolExecutor``, which would
re-import curve_curator per worker for no additional coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from drevalpy.curation._fit import (
    _build_config,
    _build_work_items,
    _fit_chunk,
    _run_work_items,
    fit_groups,
)
from drevalpy.curation._preprocess import preprocess

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


@pytest.fixture()
def single_group() -> list[tuple[pd.DataFrame, dict]]:
    """One dose-range group holding three curves."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B", "CL_C"], "DrugX")
    return preprocess(pd.DataFrame(rows))


@pytest.fixture()
def two_groups() -> list[tuple[pd.DataFrame, dict]]:
    """Two dose-range groups: DrugY tops out two decades above DrugX."""
    rows = _curve_rows([0.001, 0.01, 0.1, 1.0, 10.0], ["CL_A", "CL_B"], "DrugX")
    rows += _curve_rows([0.001, 0.01, 0.1, 1.0, 100.0], ["CL_A", "CL_B"], "DrugY")
    return preprocess(pd.DataFrame(rows))


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
        work_items, _ = _build_work_items(single_group, chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        assert len(work_items) == 2

    def test_chunks_partition_the_group(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(single_group, chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        assert sum(len(chunk) for chunk, _, _ in work_items) == len(single_group[0][0])

    def test_chunk_index_is_reset(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(single_group, chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        _, trailing_chunk = work_items[0][0], work_items[1][0]
        assert trailing_chunk.index.tolist() == [0]

    def test_one_config_per_group(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        _, configs = _build_work_items(two_groups, chunk_size=10, normalize=False, fit_type="OLS", fit_speed="fast")

        assert len(configs) == 2

    def test_each_chunk_carries_its_group_index(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(two_groups, chunk_size=10, normalize=False, fit_type="OLS", fit_speed="fast")

        assert [group_idx for _, _, group_idx in work_items] == [0, 1]


class TestFitChunk:
    """Single-chunk fitting."""

    def test_does_not_mutate_the_shared_config(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        wide_df, group_info = single_group[0]
        config = _build_config(group_info["n_experiments"], group_info["doses"], 1, fit_speed="fast")
        config["Processing"]["available_cores"] = 8

        _fit_chunk(wide_df, config)

        assert config["Processing"]["available_cores"] == 8

    def test_returns_one_row_per_curve(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        wide_df, group_info = single_group[0]
        config = _build_config(group_info["n_experiments"], group_info["doses"], 1, fit_speed="fast")

        fitted = _fit_chunk(wide_df, config)

        assert len(fitted) == len(wide_df)


class TestRunWorkItems:
    """Dispatch between the serial and the pooled fitting paths."""

    @pytest.mark.parametrize(
        ("chunk_size", "cores"),
        [
            pytest.param(2, 1, id="single-core-many-chunks"),
            pytest.param(10, 4, id="many-cores-single-chunk"),
        ],
    )
    def test_stays_in_process(
        self,
        single_group: list[tuple[pd.DataFrame, dict]],
        monkeypatch: pytest.MonkeyPatch,
        chunk_size: int,
        cores: int,
    ) -> None:
        def _no_pool(*args: object, **kwargs: object) -> None:
            raise AssertionError("ProcessPoolExecutor must not be used on the serial path")

        monkeypatch.setattr("drevalpy.curation._fit.ProcessPoolExecutor", _no_pool)
        work_items, _ = _build_work_items(
            single_group, chunk_size=chunk_size, normalize=False, fit_type="OLS", fit_speed="fast"
        )

        results = _run_work_items(work_items, cores=cores)

        assert len(results) == len(work_items)

    def test_preserves_group_index_per_chunk(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(two_groups, chunk_size=1, normalize=False, fit_type="OLS", fit_speed="fast")

        results = _run_work_items(work_items, cores=1)

        assert [group_idx for _, group_idx in results] == [group_idx for _, _, group_idx in work_items]


class TestFitGroups:
    """The public entry point."""

    def test_returns_one_result_per_group(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        results = fit_groups(two_groups, cores=1, fit_speed="fast")

        assert len(results) == 2

    def test_keeps_every_curve(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        results = fit_groups(single_group, cores=1, fit_speed="fast")

        fitted_df, _ = results[0]
        assert fitted_df["Name"].tolist() == single_group[0][0]["Name"].tolist()

    def test_routes_curves_back_to_their_own_group(self, two_groups: list[tuple[pd.DataFrame, dict]]) -> None:
        results = fit_groups(two_groups, cores=1, fit_speed="fast")

        assert [sorted(df["Name"]) for df, _ in results] == [
            ["CL_A|DrugX", "CL_B|DrugX"],
            ["CL_A|DrugY", "CL_B|DrugY"],
        ]

    def test_adds_the_curve_parameters_postprocess_consumes(
        self, single_group: list[tuple[pd.DataFrame, dict]]
    ) -> None:
        results = fit_groups(single_group, cores=1, fit_speed="fast")

        fitted_df, _ = results[0]
        assert {"pEC50", "Curve Slope", "Curve Front", "Curve Back", "Curve AUC"} <= set(fitted_df.columns)

    def test_applies_significance_thresholds(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        """Only ``thresholding.apply_significance_thresholds`` adds these columns."""
        results = fit_groups(single_group, cores=1, fit_speed="fast")

        fitted_df, _ = results[0]
        assert {"Curve Relevance Score", "Curve Regulation"} <= set(fitted_df.columns)

    def test_returns_the_config_each_group_was_fitted_with(self, single_group: list[tuple[pd.DataFrame, dict]]) -> None:
        results = fit_groups(single_group, cores=1, fit_speed="fast")

        _, config = results[0]
        assert config["Curve Fit"]["speed"] == "fast"
        assert config["Experiment"]["doses"].tolist() == single_group[0][1]["doses"]
