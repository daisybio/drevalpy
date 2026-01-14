"""Tests for the available datasets."""

from drevalpy.datasets import AVAILABLE_DATASETS


def test_factory() -> None:
    """Test the dataset factory."""
    assert "GDSC1" in AVAILABLE_DATASETS
    assert "GDSC2" in AVAILABLE_DATASETS
    assert "CCLE" in AVAILABLE_DATASETS
    assert "TOYv1" in AVAILABLE_DATASETS
    assert "TOYv2" in AVAILABLE_DATASETS
    assert "CTRPv1" in AVAILABLE_DATASETS
    assert "CTRPv2" in AVAILABLE_DATASETS
    assert "BeatAML2" in AVAILABLE_DATASETS
    assert "PDX_Bruna" in AVAILABLE_DATASETS
    assert len(AVAILABLE_DATASETS) == 9
