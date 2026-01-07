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


'''
def test_datasets():
    """Test whether the datasets exist on Zenodo."""
    zenodo_doi_url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
    expected_files = {
        "GDSC1.zip",
        "GDSC2.zip",
        "CCLE.zip",
        "CTRPv1.zip",
        "CTRPv2.zip",
        "TOYv1.zip",
        "TOYv2.zip",
        "BeatAML2.zip",
        "PDX_Bruna.zip",
        "meta.zip",
    }

    response = requests.get(zenodo_doi_url, timeout=30)
    response.raise_for_status()

    latest_url = response.links["linkset"]["url"]
    response = requests.get(latest_url, timeout=30)
    response.raise_for_status()

    data = response.json()
    zenodo_files = {f["key"] for f in data["files"]}

    missing = expected_files - zenodo_files
    assert not missing, f"Missing files on Zenodo: {missing}"
'''
