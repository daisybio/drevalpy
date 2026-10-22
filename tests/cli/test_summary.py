"""Tests for :mod:`drevalpy.cli._summary`, the grouped summary printed by ``--group-by``."""

from __future__ import annotations

import numpy as np

from drevalpy.cli._summary import GroupBy, format_grouped_summary
from drevalpy.evaluation import score_predictions
from drevalpy.types.results import ExperimentResult
from tests.synthetic import make_run_result


def _experiment() -> ExperimentResult:
    constant = make_run_result(model_name="DrugMean", n_pairs=40)
    constant.predictions = np.array([float(i % 4) for i in range(40)])
    constant.metrics = score_predictions(constant.predictions, constant.ground_truth, ["MSE", "Pearson"])
    varying = make_run_result(model_name="Varying", n_pairs=40)
    varying.metrics = score_predictions(varying.predictions, varying.ground_truth, ["MSE", "Pearson"])
    return ExperimentResult([constant, varying])


def test_the_group_by_values_are_drug_sample_and_study() -> None:
    assert [member.value for member in GroupBy] == ["drug", "sample", "study"]


def test_an_undefined_metric_is_labelled_rather_than_shown_as_a_number() -> None:
    lines = format_grouped_summary(_experiment(), GroupBy.DRUG).splitlines()

    constant = next(line for line in lines if line.startswith("DrugMean"))
    assert "undefined" in constant
    assert "nan" not in constant.lower()


def test_a_defined_metric_shows_its_mean_and_group_count() -> None:
    lines = format_grouped_summary(_experiment(), GroupBy.DRUG).splitlines()

    varying = next(line for line in lines if line.startswith("Varying"))
    assert "(4)" in varying


def test_the_heading_names_the_grouping() -> None:
    assert format_grouped_summary(_experiment(), GroupBy.SAMPLE).startswith("Metrics computed within each sample")


def test_every_model_gets_a_row_under_a_header() -> None:
    lines = format_grouped_summary(_experiment(), GroupBy.DRUG).splitlines()

    assert lines[1].split() == ["Model", "MSE", "Pearson"]
    assert [line.split()[0] for line in lines[2:]] == ["DrugMean", "Varying"]


def _prefixed_experiment() -> ExperimentResult:
    run = make_run_result(model_name="Varying", n_pairs=40, n_samples=8, n_drugs=5)
    run.sample_ids = np.array([f"{'GDSC2' if i % 2 else 'CTRPv2'}::CL_{i % 8}" for i in range(40)])
    run.metrics = score_predictions(run.predictions, run.ground_truth, ["MSE", "Pearson"])
    return ExperimentResult([run])


def test_grouping_by_study_scores_each_prefix_of_a_concatenated_run() -> None:
    lines = format_grouped_summary(_prefixed_experiment(), GroupBy.STUDY).splitlines()

    assert lines[0].startswith("Metrics computed within each study, averaged over studies and folds")
    assert "(2)" in next(line for line in lines if line.startswith("Varying"))


def test_grouping_by_study_on_a_plain_run_is_undefined() -> None:
    lines = format_grouped_summary(_experiment(), GroupBy.STUDY).splitlines()

    assert all("undefined" in line for line in lines[2:])
