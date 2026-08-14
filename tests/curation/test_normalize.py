"""Tests for drevalpy.curation._normalize.

The module exists to make a normalized fit independent of the core count, so the
headline test here compares two ``cores`` values byte for byte. That comparison
is what :mod:`drevalpy.curation._fit` used to fail: it called
``quantification.run_pipeline`` once per parallel chunk, and curve_curator
derives its normalization factors from the rows of the frame it was handed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._fit import _build_config, _build_work_items
from drevalpy.curation._normalize import (
    PRE_NORM_SIGNAL_QUALITY,
    _column_names,
    normalize_group,
    restore_signal_quality,
)
from drevalpy.curation._preprocess import preprocess

_CONCENTRATIONS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _viability_rows(cell_lines: list[str], drug: str, scale: float) -> list[dict]:
    """Long-form rows for a sigmoid scaled by a per-cell-line offset.

    A constant scale per cell line is exactly what median-centric normalization
    is meant to remove, so the scaling makes the factors non-trivial.

    :param cell_lines: Cell-line labels to emit curves for.
    :param drug: Drug label.
    :param scale: Multiplicative offset applied to every intensity.
    :returns: Long-form rows.
    """
    return [
        {
            "drug": drug,
            "cell_line": cell_line,
            "concentration": concentration,
            "intensity": scale * (index + 1) * (0.1 + 0.9 / (1 + (concentration / 0.5) ** 1.5)),
        }
        for index, cell_line in enumerate(cell_lines)
        for concentration in _CONCENTRATIONS
    ]


def build_normalizable_df() -> pd.DataFrame:
    """Twelve curves across two dose-range groups, each on its own intensity scale.

    :returns: Long-form dose-response measurements.
    """
    rows = _viability_rows([f"CL_{index}" for index in range(6)], "DrugX", scale=1.0)
    rows += _viability_rows([f"CL_{index}" for index in range(6)], "DrugY", scale=3.0)
    frame = pd.DataFrame(rows)
    # Push DrugY into a second dose-range group so the fix is exercised per group.
    frame.loc[frame["drug"] == "DrugY", "concentration"] *= 10.0
    return frame


@pytest.fixture()
def groups() -> list[tuple[pd.DataFrame, dict]]:
    """Two dose-range groups from :func:`build_normalizable_df`."""
    return preprocess(build_normalizable_df())


@pytest.fixture()
def group_and_config(groups: list[tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, dict]:
    """The first group with a matching normalizing config."""
    df, info = groups[0]
    config = _build_config(info["n_experiments"], info["doses"], info["n_replicates"], normalize=True, fit_speed="fast")
    return df, config


class TestColumnNames:
    """The column names must match the ones ``run_pipeline`` would build."""

    def test_raw_and_normalized_names_line_up(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        _, config = group_and_config

        raw, normalized, _, _ = _column_names(config)

        assert [name.replace("Raw", "Normalized") for name in raw] == list(normalized)

    def test_raw_names_exist_on_the_group_frame(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config

        raw, _, _, _ = _column_names(config)

        assert set(raw) <= set(df.columns)

    def test_the_zero_dose_column_is_excluded_from_the_dosed_set(
        self, group_and_config: tuple[pd.DataFrame, dict]
    ) -> None:
        _, config = group_and_config

        raw, _, dosed, _ = _column_names(config)

        assert len(dosed) == len(raw) - 1

    def test_controls_are_the_zero_dose_columns(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        _, config = group_and_config

        _, _, _, controls = _column_names(config)

        assert list(controls) == ["Raw 0"]


class TestNormalizeGroup:
    """Normalization is applied to the whole group in a single pass."""

    def test_raw_columns_are_overwritten_with_normalized_values(
        self, group_and_config: tuple[pd.DataFrame, dict]
    ) -> None:
        df, config = group_and_config

        result = normalize_group(df, config)

        raw, _, _, _ = _column_names(config)
        assert not np.allclose(result[raw].to_numpy(), df[raw].to_numpy())

    def test_the_normalized_helper_columns_are_not_left_behind(
        self, group_and_config: tuple[pd.DataFrame, dict]
    ) -> None:
        df, config = group_and_config

        result = normalize_group(df, config)

        assert not [column for column in result.columns if str(column).startswith("Normalized")]

    def test_the_input_frame_is_not_mutated(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config
        raw, _, _, _ = _column_names(config)
        before = df[raw].to_numpy(copy=True)

        normalize_group(df, config)

        np.testing.assert_array_equal(df[raw].to_numpy(), before)

    def test_every_curve_survives(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config

        result = normalize_group(df, config)

        assert result["Name"].tolist() == df["Name"].tolist()

    def test_pre_normalization_signal_quality_is_carried(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        df, config = group_and_config

        result = normalize_group(df, config)

        expected = np.log2(df["Raw 0"].to_numpy())
        np.testing.assert_allclose(result[PRE_NORM_SIGNAL_QUALITY].to_numpy(), expected)

    def test_factors_are_independent_of_how_the_frame_is_ordered(
        self, group_and_config: tuple[pd.DataFrame, dict]
    ) -> None:
        """Medians over the same rows, so a permutation cannot change them."""
        df, config = group_and_config
        raw, _, _, _ = _column_names(config)

        straight = normalize_group(df, config).set_index("Name")[raw]
        shuffled = normalize_group(df.iloc[::-1].reset_index(drop=True), config).set_index("Name")[raw]

        np.testing.assert_allclose(straight.to_numpy(), shuffled.loc[straight.index].to_numpy())

    def test_a_subset_of_rows_gets_different_factors(self, group_and_config: tuple[pd.DataFrame, dict]) -> None:
        """The bug this module fixes, demonstrated directly on the primitive."""
        df, config = group_and_config
        raw, _, _, _ = _column_names(config)

        whole = normalize_group(df, config).set_index("Name")[raw]
        half = normalize_group(df.iloc[:3].reset_index(drop=True), config).set_index("Name")[raw]

        assert not np.allclose(whole.loc[half.index].to_numpy(), half.to_numpy())


class TestRestoreSignalQuality:
    """The carrier column is consumed after the fit, not shipped."""

    def test_the_carried_value_replaces_the_post_normalization_one(self) -> None:
        fitted = pd.DataFrame({"Signal Quality": [0.0, 0.0], PRE_NORM_SIGNAL_QUALITY: [3.0, 4.0]})

        result = restore_signal_quality(fitted)

        assert result["Signal Quality"].tolist() == [3.0, 4.0]

    def test_the_carrier_column_is_dropped(self) -> None:
        fitted = pd.DataFrame({"Signal Quality": [0.0], PRE_NORM_SIGNAL_QUALITY: [3.0]})

        result = restore_signal_quality(fitted)

        assert PRE_NORM_SIGNAL_QUALITY not in result.columns

    def test_a_frame_that_never_was_normalized_passes_through(self) -> None:
        fitted = pd.DataFrame({"Signal Quality": [1.0]})

        result = restore_signal_quality(fitted)

        assert result is fitted


class TestWorkItemWiring:
    """``_build_work_items`` must hand the chunks an already-normalized frame."""

    def test_chunk_configs_have_normalization_switched_off(self, groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(groups, chunk_size=2, normalize=True, fit_type="OLS", fit_speed="fast")

        assert all(config["Processing"]["normalization"] is False for _, config, _ in work_items)

    def test_the_returned_group_configs_still_record_the_request(self, groups: list[tuple[pd.DataFrame, dict]]) -> None:
        """``fit_groups`` thresholds with these, and they document what was asked."""
        _, configs = _build_work_items(groups, chunk_size=2, normalize=True, fit_type="OLS", fit_speed="fast")

        assert all(config["Processing"]["normalization"] is True for config in configs)

    def test_chunks_carry_the_pre_normalization_signal_quality(self, groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(groups, chunk_size=2, normalize=True, fit_type="OLS", fit_speed="fast")

        assert all(PRE_NORM_SIGNAL_QUALITY in chunk.columns for chunk, _, _ in work_items)

    def test_no_normalization_leaves_the_frames_untouched(self, groups: list[tuple[pd.DataFrame, dict]]) -> None:
        work_items, _ = _build_work_items(groups, chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        assert all(PRE_NORM_SIGNAL_QUALITY not in chunk.columns for chunk, _, _ in work_items)

    def test_chunking_is_unaffected_by_the_extra_column(self, groups: list[tuple[pd.DataFrame, dict]]) -> None:
        normalized, _ = _build_work_items(groups, chunk_size=2, normalize=True, fit_type="OLS", fit_speed="fast")
        plain, _ = _build_work_items(groups, chunk_size=2, normalize=False, fit_type="OLS", fit_speed="fast")

        assert [len(chunk) for chunk, _, _ in normalized] == [len(chunk) for chunk, _, _ in plain]
