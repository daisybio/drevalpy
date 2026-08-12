"""Tests for the dataset identifier and measure constants."""

from __future__ import annotations

from drevalpy.data.utils import (
    ALLOWED_MEASURES,
    CELL_LINE_IDENTIFIER,
    DRUG_IDENTIFIER,
    TISSUE_IDENTIFIER,
)


class TestIdentifiers:
    """The column names datasets are keyed by."""

    def test_drug_identifier(self) -> None:
        assert DRUG_IDENTIFIER == "pubchem_id"

    def test_cell_line_identifier(self) -> None:
        assert CELL_LINE_IDENTIFIER == "cell_line_name"

    def test_tissue_identifier(self) -> None:
        assert TISSUE_IDENTIFIER == "tissue"

    def test_identifiers_are_distinct(self) -> None:
        assert len({DRUG_IDENTIFIER, CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER}) == 3


class TestAllowedMeasures:
    """``ALLOWED_MEASURES`` is extended in place with curvecurator variants."""

    def test_contains_the_base_measures(self) -> None:
        assert {"LN_IC50", "EC50", "IC50", "pEC50", "AUC", "response"} <= set(ALLOWED_MEASURES)

    def test_every_base_measure_has_a_curvecurator_variant(self) -> None:
        base = [measure for measure in ALLOWED_MEASURES if not measure.endswith("_curvecurator")]
        assert [f"{measure}_curvecurator" for measure in base] == [
            measure for measure in ALLOWED_MEASURES if measure.endswith("_curvecurator")
        ]

    def test_has_no_duplicates(self) -> None:
        assert len(ALLOWED_MEASURES) == len(set(ALLOWED_MEASURES))

    def test_extension_ran_exactly_once(self) -> None:
        """Guards against a second in-place ``extend`` on re-import."""
        assert not any(measure.endswith("_curvecurator_curvecurator") for measure in ALLOWED_MEASURES)
