"""Tests for the dataset identifier constants."""

from __future__ import annotations

from drevalpy.data.utils import CELL_LINE_NAME_COLUMN, DRUG_IDENTIFIER, TISSUE_IDENTIFIER


def test_identifiers_are_the_expected_distinct_column_names() -> None:
    assert (DRUG_IDENTIFIER, CELL_LINE_NAME_COLUMN, TISSUE_IDENTIFIER) == ("pubchem_id", "cell_line_name", "tissue")
