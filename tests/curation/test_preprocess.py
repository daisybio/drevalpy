"""Tests for drevalpy.curation._preprocess.preprocess."""

from __future__ import annotations

import pandas as pd
import pytest

from drevalpy.curation._preprocess import preprocess


class TestPreprocess:
    """Tests for drevalpy.curation._preprocess.preprocess."""

    def test_returns_list_of_tuples(self, dose_response_df: pd.DataFrame) -> None:
        result = preprocess(dose_response_df)
        assert isinstance(result, list)
        assert len(result) >= 1
        for item in result:
            assert isinstance(item, tuple) and len(item) == 2

    def test_wide_df_has_expected_columns(self, dose_response_df: pd.DataFrame) -> None:
        groups = preprocess(dose_response_df)
        wide_df, _ = groups[0]
        assert "Name" in wide_df.columns
        raw_cols = [c for c in wide_df.columns if str(c).startswith("Raw")]
        assert len(raw_cols) > 0

    def test_group_info_structure(self, dose_response_df: pd.DataFrame) -> None:
        groups = preprocess(dose_response_df)
        _, group_info = groups[0]
        assert "n_experiments" in group_info
        assert "doses" in group_info
        assert "n_replicates" in group_info
        assert isinstance(group_info["doses"], list)
        assert group_info["n_experiments"] == len([c for c in groups[0][0].columns if str(c).startswith("Raw")])

    def test_missing_columns_raises(self) -> None:
        bad_df = pd.DataFrame({"drug": ["A"], "cell_line": ["B"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(bad_df)

    def test_with_replicates(self, dose_response_df_with_replicates: pd.DataFrame) -> None:
        groups = preprocess(dose_response_df_with_replicates)
        _, group_info = groups[0]
        assert group_info["n_replicates"] == 2

    def test_name_column_format(self, dose_response_df: pd.DataFrame) -> None:
        groups = preprocess(dose_response_df)
        wide_df, _ = groups[0]
        for name in wide_df["Name"]:
            parts = name.split("|")
            assert len(parts) == 2

    def test_duplicate_measurements_are_averaged_with_a_warning(self, dose_response_df: pd.DataFrame) -> None:
        duplicated = pd.concat([dose_response_df, dose_response_df.head(1)], ignore_index=True)

        with pytest.warns(UserWarning, match="Duplicate entries found"):
            groups = preprocess(duplicated)

        wide_df, _ = groups[0]
        assert len(wide_df) == 6
