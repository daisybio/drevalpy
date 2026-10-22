"""Tests for the dataset identifier constants."""

from __future__ import annotations

from drevalpy.data.utils import (
    CELL_LINE_NAME_COLUMN,
    DRUG_IDENTIFIER,
    TISSUE_IDENTIFIER,
)


class TestIdentifiers:
    """The column names datasets are keyed by."""

    def test_drug_identifier(self) -> None:
        assert DRUG_IDENTIFIER == "pubchem_id"

    def test_sample_identifier(self) -> None:
        assert CELL_LINE_NAME_COLUMN == "cell_line_name"

    def test_tissue_identifier(self) -> None:
        assert TISSUE_IDENTIFIER == "tissue"

    def test_identifiers_are_distinct(self) -> None:
        assert len({DRUG_IDENTIFIER, CELL_LINE_NAME_COLUMN, TISSUE_IDENTIFIER}) == 3
