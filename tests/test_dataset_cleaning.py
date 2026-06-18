"""Tests for the generic drug-cleaning mechanism (DrugCurveFilter + derived datasets)."""

import pandas as pd
import pytest

from drevalpy.datasets import AVAILABLE_DATASETS
from drevalpy.datasets.loader import DERIVED_DATASETS, DrugCurveFilter, register_clean_tiers


def _toy_frame() -> pd.DataFrame:
    """Synthetic curve-curated response frame: drug A has 3 significant curves, B has 1, C has 2."""
    return pd.DataFrame(
        {
            "pubchem_id": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "cell_line_name": list("abcd") + list("abcd") + ["a", "b"],
            "Regulation": ["down", "up", "down", "-", "-", "-", "-", "up", "down", "up"],
            "LN_IC50_curvecurator": range(10),
        }
    )


def test_filter_absolute_count() -> None:
    """min_responders keeps only drugs with >= N significant curves (A:3, B:1, C:2)."""
    kept = set(DrugCurveFilter(min_responders=3).apply(_toy_frame())["pubchem_id"])
    assert kept == {"A"}
    kept2 = set(DrugCurveFilter(min_responders=2).apply(_toy_frame())["pubchem_id"])
    assert kept2 == {"A", "C"}


def test_filter_fraction() -> None:
    """min_responder_frac keeps drugs whose significant fraction meets the threshold (A:3/4, B:1/4, C:2/2)."""
    kept = set(DrugCurveFilter(min_responder_frac=0.5).apply(_toy_frame())["pubchem_id"])
    assert kept == {"A", "C"}


def test_filter_requires_exactly_one_criterion() -> None:
    """Setting neither or both of the two thresholds is a configuration error."""
    with pytest.raises(ValueError):
        DrugCurveFilter()
    with pytest.raises(ValueError):
        DrugCurveFilter(min_responders=5, min_responder_frac=0.1)


def test_filter_requires_curve_curated() -> None:
    """A frame without the CurveCurator 'Regulation' column cannot be cleaned."""
    frame = _toy_frame().drop(columns=["Regulation"])
    with pytest.raises(ValueError, match="Regulation"):
        DrugCurveFilter(min_responders=1).apply(frame)


def test_ctrpv2_tiers_registered() -> None:
    """The three CTRPv2 clean tiers are registered and available, with absolute thresholds."""
    for name, expected in [("CTRPv2_clean", 15), ("CTRPv2_cleaner", 30), ("CTRPv2_cleanest", 50)]:
        assert name in AVAILABLE_DATASETS
        base, filt = DERIVED_DATASETS[name]
        assert base == "CTRPv2"
        assert filt.min_responders == expected


def test_register_clean_tiers_is_general() -> None:
    """register_clean_tiers works for any base dataset, not just CTRPv2."""
    added = register_clean_tiers("GDSC2", {"GDSC2_clean_test": 20})
    try:
        assert DERIVED_DATASETS["GDSC2_clean_test"] == ("GDSC2", added["GDSC2_clean_test"])
        assert added["GDSC2_clean_test"].min_responders == 20
    finally:
        DERIVED_DATASETS.pop("GDSC2_clean_test", None)
