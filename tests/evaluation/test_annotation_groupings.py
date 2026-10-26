"""Per-group scores by a sample or drug annotation column of a dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.data import synthetic
from drevalpy.evaluation import Metric, PerGroupScores, evaluate_per_group
from drevalpy.evaluation import _grouping as grouping_module
from drevalpy.evaluation._grouping import defined_labels, group_labeller, group_labels
from drevalpy.evaluation._scoring import mean_of_defined
from drevalpy.types import Dataset
from drevalpy.types.results import ExperimentResult, ModelResult, RunResult
from tests.synthetic import make_run_result

N_SAMPLES = 6
N_DRUGS = 4
SAMPLE_IDS = [f"SYNTH-{index:03d}" for index in range(N_SAMPLES)]
DRUG_IDS = [str(100000 + index) for index in range(N_DRUGS)]
TISSUES = ["Lung", "Blood", "Skin", "Lung", "Blood", "Skin"]
CANCER_TYPES = ["a", "a", "b", "b", "unknown", "UNKNOWN"]
DRUG_CLASSES = ["x", "x", "y", "unknown"]


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Six samples and four drugs with annotations that cover every kind of unusable value."""
    data = synthetic(n_samples=N_SAMPLES, n_drugs=N_DRUGS, n_tissues=3)
    data.mdata.obs["cancer_type"] = CANCER_TYPES
    data.mdata.obs["level"] = [1, 1, 2, 2, 3, 3]
    data.mdata.obs["note"] = ["p", None, "", "  ", "q", np.nan]
    data.response.var["ncit_class"] = DRUG_CLASSES
    data.mdata.obs["origin"] = ["s0", "s0", "s1", "s1", "s2", "s2"]
    data.response.var["origin"] = ["d0", "d0", "d1", "d1"]
    return data


def _run(model_name: str = "ElasticNet", seed: int = 0, extra_sample_id: str | None = None) -> RunResult:
    """A run over every sample-drug pair of the dataset, plus one pair of an ID the dataset lacks."""
    run = make_run_result(model_name=model_name, n_pairs=N_SAMPLES * N_DRUGS, seed=seed)
    run.sample_ids = np.array([SAMPLE_IDS[i % N_SAMPLES] for i in range(run.predictions.size)])
    run.drug_ids = np.array([DRUG_IDS[i // N_SAMPLES] for i in range(run.predictions.size)])
    if extra_sample_id is not None:
        run.sample_ids[0] = extra_sample_id
    return run


def _listed(labels: np.ndarray) -> list[str | None]:
    """The labels as a list with ``None`` for no group, which compares equal unlike NaN."""
    return [label if isinstance(label, str) else None for label in labels]


def _expected(run: RunResult, labels: list[object]) -> PerGroupScores:
    """Per-group scores of hand-built labels (NaN for no group)."""
    return evaluate_per_group(run.predictions, run.ground_truth, labels=np.array(labels, dtype=object))


class TestBuiltinsNeedNoDataset:
    def test_drug_sample_and_study_are_unchanged(self, dataset: Dataset) -> None:
        run = _run()
        run.sample_ids = np.array([f"GDSC2::{sample}" for sample in run.sample_ids])

        for grouping in ("drug", "sample", "study"):
            without = group_labels(run, grouping)
            np.testing.assert_array_equal(group_labels(run, grouping, dataset), without)
        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)
        np.testing.assert_array_equal(group_labels(run, "sample"), run.sample_ids)
        assert set(group_labels(run, "study")) == {"GDSC2"}

    def test_a_builtin_is_not_looked_up_in_the_dataset(self, dataset: Dataset) -> None:
        run = _run(extra_sample_id="GHOST")

        assert group_labels(run, "sample", dataset)[0] == "GHOST"


class TestAnnotationLabels:
    def test_a_sample_annotation_labels_the_pairs_of_each_sample(self, dataset: Dataset) -> None:
        run = _run()

        labels = group_labels(run, "tissue", dataset)

        assert _listed(labels) == [TISSUES[i % N_SAMPLES] for i in range(run.predictions.size)]

    def test_per_group_scores_match_hand_built_labels(self, dataset: Dataset) -> None:
        run = _run()
        hand_built = [TISSUES[i % N_SAMPLES] for i in range(run.predictions.size)]

        scores = evaluate_per_group(run.predictions, run.ground_truth, labels=group_labels(run, "tissue", dataset))
        expected = _expected(run, hand_built)

        assert scores.groups.tolist() == expected.groups.tolist() == ["Blood", "Lung", "Skin"]
        for name in scores.values:
            np.testing.assert_array_equal(scores.values[name], expected.values[name])

    def test_a_drug_annotation_labels_the_pairs_of_each_drug(self, dataset: Dataset) -> None:
        run = _run()

        labels = group_labels(run, "ncit_class", dataset)

        by_drug = dict(zip(DRUG_IDS, DRUG_CLASSES, strict=True))
        expected = [by_drug[drug] for drug in run.drug_ids]
        assert _listed(labels) == [None if value == "unknown" else value for value in expected]
        scores = evaluate_per_group(run.predictions, run.ground_truth, labels=labels)
        hand_built = [None if value == "unknown" else value for value in expected]
        expected_scores = _expected(run, [np.nan if value is None else value for value in hand_built])
        assert scores.groups.tolist() == expected_scores.groups.tolist() == ["x", "y"]
        for name in scores.values:
            np.testing.assert_array_equal(scores.values[name], expected_scores.values[name])

    def test_non_string_values_are_converted_to_strings(self, dataset: Dataset) -> None:
        run = _run()

        labels = group_labels(run, "level", dataset)

        assert set(labels.tolist()) == {"1", "2", "3"}

    def test_unknown_missing_and_empty_labels_belong_to_no_group(self, dataset: Dataset) -> None:
        run = _run()

        labels = group_labels(run, "cancer_type", dataset)

        assert defined_labels(labels).tolist() == [i % N_SAMPLES < 4 for i in range(run.predictions.size)]
        assert set(labels[defined_labels(labels)]) == {"a", "b"}
        noted = group_labels(run, "note", dataset)
        assert set(noted[defined_labels(noted)]) == {"p", "q"}

    def test_the_unknown_marker_is_case_insensitive(self, dataset: Dataset) -> None:
        run = _run()

        labels = group_labels(run, "cancer_type", dataset)

        assert not defined_labels(labels)[[4, 5]].any()

    def test_a_pair_whose_id_is_not_in_the_dataset_belongs_to_no_group(self, dataset: Dataset) -> None:
        run = _run(extra_sample_id="GHOST")

        labels = group_labels(run, "tissue", dataset)

        assert not defined_labels(labels)[0]
        assert defined_labels(labels).sum() == run.predictions.size - 1
        drug_run = _run()
        drug_run.drug_ids[0] = "999"
        assert not defined_labels(group_labels(drug_run, "ncit_class", dataset))[0]

    def test_a_dataset_with_no_matching_ids_leaves_no_groups(self, dataset: Dataset) -> None:
        run = _run()
        run.sample_ids = np.array([f"OTHER-{i}" for i in range(run.predictions.size)])

        scores = evaluate_per_group(run.predictions, run.ground_truth, labels=group_labels(run, "tissue", dataset))

        assert scores.groups.size == 0
        assert all(values.size == 0 for values in scores.values.values())


class TestAxisSelection:
    def test_the_sample_prefix_selects_the_sample_axis(self, dataset: Dataset) -> None:
        run = _run()

        for prefixed, bare in (("sample:cancer_type", "cancer_type"), ("drug:ncit_class", "ncit_class")):
            assert _listed(group_labels(run, prefixed, dataset)) == _listed(group_labels(run, bare, dataset))

    def test_a_name_on_both_axes_needs_a_prefix(self, dataset: Dataset) -> None:
        run = _run()

        with pytest.raises(ValueError, match=r"'sample:origin'.*'drug:origin'"):
            group_labels(run, "origin", dataset)
        assert set(group_labels(run, "sample:origin", dataset)) == {"s0", "s1", "s2"}
        assert set(group_labels(run, "drug:origin", dataset)) == {"d0", "d1"}

    def test_a_prefix_restricts_the_search_to_its_axis(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="Unknown grouping 'drug:cancer_type'"):
            group_labels(_run(), "drug:cancer_type", dataset)

    def test_an_unknown_column_lists_the_annotations_of_both_axes(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="Unknown grouping 'nope'") as error:
            group_labels(_run(), "nope", dataset)

        message = str(error.value)
        assert "Sample annotations:" in message
        assert "'cancer_type'" in message
        assert "Drug annotations:" in message
        assert "'ncit_class'" in message

    def test_an_annotation_without_a_dataset_asks_for_one(self) -> None:
        with pytest.raises(ValueError, match="Pass dataset="):
            group_labels(_run(), "tissue")
        with pytest.raises(ValueError, match="dataset="):
            group_labeller("sample:tissue")


class TestScoringWithADataset:
    def test_aggregate_group_metrics_matches_hand_built_labels(self, dataset: Dataset) -> None:
        runs = [_run(seed=seed) for seed in range(3)]
        model = ModelResult("ElasticNet", "SyntheticDataset", runs)
        hand_built = [TISSUES[i % N_SAMPLES] for i in range(runs[0].predictions.size)]

        result = model.aggregate_group_metrics("sample:tissue", dataset)

        for name in Metric:
            per_run = [_expected(run, hand_built) for run in runs]
            means = [scores.mean()[name] for scores in per_run]
            counts = [scores.n_defined()[name] for scores in per_run]
            assert result[name]["mean"] == pytest.approx(mean_of_defined(means))
            assert result[name]["n_groups"] == pytest.approx(np.mean(counts))

    def test_the_annotation_is_resolved_once_per_call_not_per_run(
        self, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = ModelResult("ElasticNet", "SyntheticDataset", [_run(seed=seed) for seed in range(4)])
        calls: list[int] = []
        original = grouping_module._clean_annotation

        def counting(column):
            calls.append(1)
            return original(column)

        monkeypatch.setattr(grouping_module, "_clean_annotation", counting)

        model.aggregate_group_metrics("cancer_type", dataset)

        assert len(calls) == 1

    def test_summary_table_and_counts_accept_a_dataset(self, dataset: Dataset) -> None:
        experiment = ExperimentResult(
            [_run(name, seed=index) for index, name in enumerate(("ElasticNet", "RandomForest"))]
        )

        table = experiment.summary_table(group_by="ncit_class", dataset=dataset)
        counts = experiment.summary_group_counts("ncit_class", dataset)

        assert list(table.index) == ["ElasticNet", "RandomForest"]
        assert table.notna().all().any()
        assert counts["MSE"].tolist() == [2.0, 2.0]
        assert experiment.summary_group_counts("cancer_type", dataset)["MSE"].tolist() == [2.0, 2.0]
        assert experiment.summary_group_counts("sample:tissue", dataset)["MSE"].tolist() == [3.0, 3.0]

    def test_the_builtin_summary_does_not_depend_on_the_dataset(self, dataset: Dataset) -> None:
        experiment = ExperimentResult([_run()])

        for grouping in ("drug", "sample", "study"):
            with_dataset = experiment.summary_table(group_by=grouping, dataset=dataset)
            without = experiment.summary_table(group_by=grouping)
            pd.testing.assert_frame_equal(with_dataset, without)

    def test_summary_table_without_a_dataset_rejects_an_annotation(self) -> None:
        experiment = ExperimentResult([_run()])

        with pytest.raises(ValueError, match="dataset="):
            experiment.summary_table(group_by="tissue")
        with pytest.raises(ValueError, match="dataset="):
            experiment.summary_group_counts("tissue")

    def test_an_invalid_grouping_is_rejected_even_without_runs(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="Unknown grouping 'nope'"):
            ModelResult("ElasticNet", "SyntheticDataset", []).aggregate_group_metrics("nope", dataset)
