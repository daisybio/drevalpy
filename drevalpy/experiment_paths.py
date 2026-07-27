"""Shared result-directory layout for experiments and CLI consolidation."""

from __future__ import annotations

from pathlib import Path


def experiment_result_path(
    path_out: str | Path,
    run_id: str,
    dataset_name: str,
    split_label: str,
) -> Path:
    """
    Return ``path_out/run_id/dataset_name/split_label``.

    :param path_out: experiment output root
    :param run_id: run identifier
    :param dataset_name: dataset name
    :param split_label: split / test-mode label
    :returns: hierarchical result path
    """
    return Path(path_out) / run_id / dataset_name / split_label


def consolidate_results_path(
    outdir_path: str | Path,
    run_id: str,
    dataset_name: str,
    test_mode: str,
) -> Path:
    """
    Return the experiment result directory used by consolidate CLI steps.

    :param outdir_path: experiment output root
    :param run_id: run identifier
    :param dataset_name: dataset name
    :param test_mode: split / test-mode label
    :returns: hierarchical result path
    """
    return experiment_result_path(outdir_path, run_id, dataset_name, test_mode)
