"""Shared result-directory layout for experiments and CLI consolidation."""

from __future__ import annotations

from upath import UPath as Path


def experiment_result_path(
    path_out: str | Path,
    run_id: str,
    dataset_name: str,
    split_label: str,
) -> Path:
    """Return ``path_out/run_id/dataset_name/split_label``.

    :param path_out: Experiment output root.
    :param run_id: Run identifier.
    :param dataset_name: Dataset name.
    :param split_label: Split or test-mode label.

    :returns: Hierarchical result path.
    """
    return Path(path_out) / run_id / dataset_name / split_label


def consolidate_results_path(
    outdir_path: str | Path,
    run_id: str,
    dataset_name: str,
    test_mode: str,
) -> Path:
    """Return the experiment result directory used by consolidate CLI steps.

    :param outdir_path: Experiment output root.
    :param run_id: Run identifier.
    :param dataset_name: Dataset name.
    :param test_mode: Split or test-mode label.

    :returns: Hierarchical result path.
    """
    return experiment_result_path(outdir_path, run_id, dataset_name, test_mode)
