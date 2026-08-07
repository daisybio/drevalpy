"""Tests for cv_splits_load."""

from __future__ import annotations

from pathlib import Path

from drevalpy.datasets.cv_splits_load import _partition_split_filenames, load_cv_splits_from_dir

HEADER = "cell_line_name,pubchem_id,response\n"


def test_load_cv_splits_from_dir_train_test(tmp_path: Path) -> None:
    (tmp_path / "cv_split_0_train.csv").write_text(HEADER + "a,d,1\n")
    (tmp_path / "cv_split_0_test.csv").write_text(HEADER + "b,d,2\n")
    splits = load_cv_splits_from_dir(str(tmp_path), "testset")
    assert len(splits) == 1
    assert len(splits[0]["train"]) == 1
    assert len(splits[0]["test"]) == 1


def test_load_cv_splits_accepts_a_path(tmp_path: Path) -> None:
    """The public entry point takes a ``Path`` as well as a ``str``.

    :param tmp_path: Temporary directory holding the split CSV files.
    """
    (tmp_path / "cv_split_0_train.csv").write_text(HEADER + "a,d,1\n")
    (tmp_path / "cv_split_0_test.csv").write_text(HEADER + "b,d,2\n")
    splits = load_cv_splits_from_dir(tmp_path, "testset")
    assert len(splits) == 1


def test_partition_pairs_folds_by_sorted_filename() -> None:
    """Train and test lists must be sorted so ``zip`` pairs matching fold indices.

    Directory listing order is arbitrary, so the sort inside
    ``_partition_split_filenames`` is what guarantees fold *i* train is paired
    with fold *i* test.
    """
    files = [
        "cv_split_2_test.csv",
        "cv_split_0_train.csv",
        "cv_split_1_test.csv",
        "cv_split_2_train.csv",
        "cv_split_0_test.csv",
        "cv_split_1_train.csv",
    ]
    partitions = _partition_split_filenames(files)
    assert partitions["train"] == [
        "cv_split_0_train.csv",
        "cv_split_1_train.csv",
        "cv_split_2_train.csv",
    ]
    assert partitions["test"] == [
        "cv_split_0_test.csv",
        "cv_split_1_test.csv",
        "cv_split_2_test.csv",
    ]
    for train, test in zip(partitions["train"], partitions["test"], strict=True):
        assert train.split("_")[2] == test.split("_")[2]


def test_partition_keeps_validation_and_validation_es_disjoint() -> None:
    """``validation_es`` files must not leak into the plain ``validation`` bucket."""
    files = [
        "cv_split_0_validation.csv",
        "cv_split_0_validation_es.csv",
        "cv_split_1_validation.csv",
        "cv_split_1_validation_es.csv",
    ]
    partitions = _partition_split_filenames(files)
    assert partitions["validation"] == ["cv_split_0_validation.csv", "cv_split_1_validation.csv"]
    assert partitions["validation_es"] == ["cv_split_0_validation_es.csv", "cv_split_1_validation_es.csv"]
    assert not set(partitions["validation"]) & set(partitions["validation_es"])
