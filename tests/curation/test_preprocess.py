"""Tests for drevalpy.curation._preprocess.preprocess."""

from __future__ import annotations

import pandas as pd
import pytest

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
