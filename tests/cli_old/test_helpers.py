"""Tests for drevalpy.cli._helpers."""

from __future__ import annotations

from drevalpy.cli._helpers import normalize_list_argv


def test_normalize_list_argv_expands_space_separated_values() -> None:
    argv = ["--models", "Simple", "ElasticNet", "--dataset_name", "GDSC1"]
    assert normalize_list_argv(argv) == [
        "--models",
        "Simple",
        "--models",
        "ElasticNet",
        "--dataset_name",
        "GDSC1",
    ]


def test_normalize_list_argv_preserves_repeated_flags() -> None:
    argv = ["evaluate-hpams", "--hpam_yamls", "a.yml", "--hpam_yamls", "b.yml"]
    assert normalize_list_argv(argv) == [
        "evaluate-hpams",
        "--hpam_yamls",
        "a.yml",
        "--hpam_yamls",
        "b.yml",
    ]


def test_normalize_list_argv_stops_at_next_option() -> None:
    argv = ["--test_mode", "LPO", "LCO", "--path_out", "results/"]
    assert normalize_list_argv(argv) == [
        "--test_mode",
        "LPO",
        "--test_mode",
        "LCO",
        "--path_out",
        "results/",
    ]


def test_normalize_list_argv_does_not_expand_scalar_subcommand_test_mode() -> None:
    argv = ["test-cv", "--test_mode", "LPO", "LCO", "--split_id", "0"]
    assert normalize_list_argv(argv) == argv


def test_normalize_list_argv_expands_subcommand_cross_study_datasets() -> None:
    argv = ["test-cv", "--cross_study_datasets", "a.pkl", "b.pkl", "--split_id", "0"]
    assert normalize_list_argv(argv) == [
        "test-cv",
        "--cross_study_datasets",
        "a.pkl",
        "--cross_study_datasets",
        "b.pkl",
        "--split_id",
        "0",
    ]


def test_normalize_list_argv_does_not_expand_scalar_randomization_mode() -> None:
    argv = ["make-randomization-yamls", "--randomization_mode", "SVCC", "SVRC", "--model_name", "Simple"]
    assert normalize_list_argv(argv) == argv
