"""Tests for the generic drug-cleaning mechanism (DrugCurveFilter + derived datasets)."""

import shutil

import pandas as pd
import pytest

from drevalpy.datasets import AVAILABLE_DATASETS
from drevalpy.datasets.loader import DERIVED_DATASETS, DrugCurveFilter, load_dataset, register_clean_tiers


def _toy_frame() -> pd.DataFrame:
    """Synthetic curve-curated response frame: drug A has 3 significant curves, B has 1, C has 2.

    :returns: a small curve-curated response frame for the cleaning tests.
    """
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


def test_load_dataset_clean_min_responders(data_dir, tmp_path) -> None:
    """load_dataset(clean_min_responders=N) derives and drug-filters any curve-curated base on the fly."""
    base_csv = data_dir / "TOYv1" / "TOYv1.csv"
    if not base_csv.is_file():
        pytest.skip("TOYv1 toy data not available")

    # hermetic copy so the derived variant is not written into the shared data dir
    shutil.copytree(data_dir / "TOYv1", tmp_path / "TOYv1")
    if (data_dir / "meta").is_dir():
        shutil.copytree(data_dir / "meta", tmp_path / "meta")

    measure = "LN_IC50_curvecurator"
    base_drugs = set(pd.read_csv(base_csv, dtype={"pubchem_id": str})["pubchem_id"])
    try:
        cleaned = load_dataset(
            dataset_name="TOYv1", path_data=str(tmp_path), measure=measure, clean_min_responders=5
        )
        kept = {str(d) for d in cleaned.drug_ids}
        looser = {
            str(d)
            for d in load_dataset(
                dataset_name="TOYv1", path_data=str(tmp_path), measure=measure, clean_min_responders=1
            ).drug_ids
        }

        assert kept <= base_drugs  # cleaning only drops whole drugs, never adds
        assert "TOYv1_clean_min5" in DERIVED_DATASETS  # variant registered on the fly
        assert (tmp_path / "TOYv1_clean_min5").is_dir()  # materialized with shared features
        assert kept <= looser  # stricter threshold keeps no more drugs than a looser one
    finally:
        DERIVED_DATASETS.pop("TOYv1_clean_min5", None)
        DERIVED_DATASETS.pop("TOYv1_clean_min1", None)
