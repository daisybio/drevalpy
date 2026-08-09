"""Regression tests for the CurveCurator preprocessing contract.

``configlist.txt`` is consumed by the external CurveCurator binary, so its exact
text content is a cross-process contract that must survive refactors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.data.curvecurator import preprocess


def _viability_frame() -> pd.DataFrame:
    rows = []
    for drug in ("drugA", "drugB"):
        for dose in (0.01, 0.1, 1.0, 10.0):
            rows.append({"dose": dose, "response": 1.0 - dose / 20.0, "sample": "cl1", "drug": drug, "replicate": 1})
    return pd.DataFrame(rows)


def test_configlist_lists_absolute_config_paths_one_per_line(tmp_path: Path) -> None:
    """Each line is the plain path of a written ``config.toml``, newline-terminated.

    :param tmp_path: Temporary directory for the input CSV and CurveCurator output.
    """
    input_file = tmp_path / "viability.csv"
    _viability_frame().to_csv(input_file, index=False)
    output_dir = tmp_path / "curvecurator"
    output_dir.mkdir()

    preprocess(input_file=input_file, output_dir=output_dir, dataset_name="TESTSET", cores=1)

    configlist = output_dir / "configlist.txt"
    lines = configlist.read_text().splitlines(keepends=True)
    assert lines, "configlist.txt must not be empty"
    for line in lines:
        assert line.endswith("\n")
        config_path = Path(line.rstrip("\n"))
        assert config_path.is_file(), f"{config_path} listed in configlist.txt does not exist"
        assert config_path.name == "config.toml"
        # No quoting, no repr(), no trailing separator: the binary reads this verbatim.
        assert line.rstrip("\n") == str(config_path)


def test_preprocess_accepts_string_paths(tmp_path: Path) -> None:
    """The public entry point still accepts plain strings.

    :param tmp_path: Temporary directory for the input CSV and CurveCurator output.
    """
    input_file = tmp_path / "viability.csv"
    _viability_frame().to_csv(input_file, index=False)
    output_dir = tmp_path / "curvecurator"
    output_dir.mkdir()

    preprocess(input_file=str(input_file), output_dir=str(output_dir), dataset_name="TESTSET", cores=1)

    assert (output_dir / "configlist.txt").is_file()
