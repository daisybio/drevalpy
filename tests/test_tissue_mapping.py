"""Test suite for tissue mapping functionality in the drevalpy package."""

import pandas as pd
import pytest

from drevalpy.datasets.map_tissues import main


@pytest.fixture
def test_data(tmp_path):
    """Create a temporary directory with dummy data for testing.

    :param tmp_path: pytest fixture for creating temporary directories.
    :returns: Tuple containing the root directory and dataset name.
    """
    # Setup directory
    root = tmp_path / "data"
    root.mkdir()
    meta = root / "meta"
    meta.mkdir()

    # Create dummy dataset
    ds_name = "TESTSET"
    ds_path = root / ds_name
    ds_path.mkdir()
    df = pd.DataFrame({"cellosaurus_id": ["CVCL_TEST1", "CVCL_TEST2"]})
    df.to_csv(ds_path / f"{ds_name}.csv", index=False)

    # Create dummy depmap metadata
    depmap = pd.DataFrame(
        {
            "DepMap_ID": ["ACH-000001", "ACH-000002"],
            "stripped_cell_line_name": ["testcl1", "testcl2"],
            "disease": ["lung", ""],
            "disease_sutype": ["lung cancer", "leukemia"],
            "disease_sub_subtype": ["adenocarcinoma", "AML"],
            "culture_type": ["Adherent", "Suspension"],
            "culture_medium": ["RPMI", "DMEM"],
            "gender": ["Male", "Female"],
            "source": ["Broad", "Broad"],
        }
    )

    depmap.to_csv(meta / "DepMap_sample_info.csv", index=False)

    # Create dummy Cellosaurus file
    cellosaurus_text = (
        "ID   TestCL1\n"
        "AC   CVCL_TEST1;\n"
        "CC   Derived from site: lung; lung\n"
        "DI   ; NCIt; Lung adenocarcinoma\n"
        "//\n"
        "ID   TestCL2\n"
        "AC   CVCL_TEST2;\n"
        "CC   Derived from site: blood; blood\n"
        "DI   ; NCIt; Acute myeloid leukemia\n"
        "//\n"
    )
    (meta / "cellosaurus.txt").write_text(cellosaurus_text, encoding="utf-8")

    return root, ds_name


def test_map_tissues(monkeypatch, test_data):
    """Test the map_tissues function.

    :param monkeypatch: pytest fixture to modify the environment for testing.
    :param test_data: fixture providing a temporary directory with dummy data.
    """
    root, ds_name = test_data

    monkeypatch.setattr("sys.argv", ["script.py", str(root), ds_name, "--save_tissue_mapping"])
    main()

    # Check output file exists
    output_path = root / "meta" / "tissue_mapping.csv"
    assert output_path.exists()

    # Load and check contents
    df_out = pd.read_csv(output_path)
    print("\n=== tissue_mapping.csv ===")
    print(df_out)

    assert "tissue" in df_out.columns
    print("\n=== tissue for CVCL_TEST1 ===")
    print(df_out.loc[df_out["cellosaurus_id"] == "CVCL_TEST1", "tissue"])

    print("\n=== tissue for CVCL_TEST2 ===")
    print(df_out.loc[df_out["cellosaurus_id"] == "CVCL_TEST2", "tissue"])

    assert df_out.loc[df_out["cellosaurus_id"] == "CVCL_TEST1", "tissue"].values[0] == "Lung"
    assert df_out.loc[df_out["cellosaurus_id"] == "CVCL_TEST2", "tissue"].values[0] == "Blood"

    # Check the updated dataset has a tissue column
    updated = pd.read_csv(root / ds_name / f"{ds_name}.csv")
    print("\n=== Updated dataset ===")
    print(updated)

    assert "tissue" in updated.columns
    assert set(updated["tissue"]) == {"Lung", "Blood"}
