"""Tests for drevalpy.curation._normalize (a normalized fit must not depend on the core count)."""

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
