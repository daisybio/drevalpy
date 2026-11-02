"""Choosing the response measure: the dataset view, splits, runs, fitted models and predictions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from upath import UPath

import drevalpy
from drevalpy._predict import predict_observed
from drevalpy.data import CurveQuality, split, synthetic
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.experiment import ExperimentResult


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A synthetic dataset whose AUC layer lacks a few pairs that have a pEC50."""
    data = synthetic(name="MEASURES", n_samples=12, n_drugs=5, extra_layers=("AUC",), seed=3)
    auc = np.array(data.response.layers["AUC"], dtype=np.float32)
    measured = np.argwhere(np.isfinite(auc))
    for i, j in measured[:6]:
        auc[i, j] = np.nan
    data.response.layers["AUC"] = auc
    return data


class TestDatasetView:
    def test_x_is_the_default_measure(self, dataset: Dataset) -> None:
        assert (dataset.measure, dataset.x_measure) == ("pEC50", "pEC50")
        assert dataset.measures()[:1] == ["pEC50"]
        assert {"LN_IC50", "AUC"} <= set(dataset.measures())

    def test_with_measure_reads_the_layer_and_shares_the_data(self, dataset: Dataset) -> None:
        auc = dataset.with_measure("AUC")

        assert auc.measure == "AUC"
        assert auc.mdata is dataset.mdata
        np.testing.assert_array_equal(auc.response_matrix, dataset.get_response_layer("AUC"))
        np.testing.assert_array_equal(auc.get_response_layer("pEC50"), dataset.response_matrix)
        assert dataset.measure == "pEC50"
        assert auc.with_measure("AUC") is auc
        assert "Measure: AUC" in repr(auc)

    def test_an_unknown_measure_lists_the_available_ones(self, dataset: Dataset) -> None:
        with pytest.raises(UnknownNameError, match=r"no response measure 'auc'\. Available: \['pEC50'"):
            dataset.with_measure("auc")

    def test_a_randomized_copy_keeps_the_measure(self, dataset: Dataset) -> None:
        randomized = dataset.with_measure("LN_IC50").with_randomized_views(["gene_expression"], random_state=1)

        assert randomized.measure == "LN_IC50"


class TestSplit:
    def test_only_pairs_with_the_measure_are_split_and_it_is_recorded(self, dataset: Dataset) -> None:
        folds = split(dataset, "LPO", n_splits=2, quality=CurveQuality.off(), measure="AUC")
        auc = np.isfinite(dataset.get_response_layer("AUC"))

        for fold in folds:
            assert fold.metadata["measure"] == "AUC"
            used = fold.train.mask | fold.val.mask | fold.test.mask
            assert not (used & ~auc).any()
        covered = np.logical_or.reduce([fold.test.mask for fold in folds])
        np.testing.assert_array_equal(covered, auc)

    def test_the_quality_rule_keeps_reading_pec50(self, dataset: Dataset) -> None:
        rule = dataclasses.replace(CurveQuality.off(), min_pec50=6.0)
        pec50 = dataset.response_matrix

        mask = rule.mask(dataset.with_measure("LN_IC50"))

        np.testing.assert_array_equal(mask, (pec50 >= 6.0) & ~np.isnan(pec50))


class TestRunAndSingle:
    def test_a_run_trains_and_scores_on_the_measure(self, dataset: Dataset) -> None:
        result = drevalpy.run(
            [ModelSpec.from_name("NaiveDrugMeanPredictor")],
            dataset,
            "LPO",
            hyperparameter_tuning=False,
            quality=CurveQuality.off(),
            measure="LN_IC50",
        )
        layer = dataset.get_response_layer("LN_IC50")
        sample_index = {sample: i for i, sample in enumerate(dataset.sample_ids)}
        drug_index = {drug: j for j, drug in enumerate(dataset.drug_ids)}
        run = result.models[0].runs[0]

        assert result.measure == "LN_IC50"
        assert run.measure == "LN_IC50"
        expected = [layer[sample_index[s], drug_index[d]] for s, d in zip(run.sample_ids, run.drug_ids, strict=True)]
        np.testing.assert_allclose(run.ground_truth, expected)

    def test_single_takes_the_measure_of_the_fold_and_rejects_another(self, dataset: Dataset) -> None:
        fold = split(dataset, "LPO", n_splits=2, quality=CurveQuality.off(), measure="AUC")[0]
        spec = ModelSpec.from_name("NaiveDrugMeanPredictor")

        result = drevalpy.single(spec, dataset, fold, hyperparameter_tuning=False)

        assert result.measure == "AUC"
        with pytest.raises(InvalidInputError, match="split on the response measure 'AUC'"):
            drevalpy.single(spec, dataset, fold, hyperparameter_tuning=False, measure="LN_IC50")

    def test_results_of_different_measures_do_not_mix(self, dataset: Dataset) -> None:
        spec = ModelSpec.from_name("NaiveDrugMeanPredictor")
        runs = [
            drevalpy.single(spec, dataset, fold, hyperparameter_tuning=False)
            for fold in (
                split(dataset, "LPO", n_splits=2, quality=CurveQuality.off(), measure=measure)[0]
                for measure in ("AUC", "LN_IC50")
            )
        ]

        assert runs[0].fold_key != runs[1].fold_key
        with pytest.raises(InvalidInputError, match="same response measure"):
            ExperimentResult(runs)


class TestFittedModel:
    @pytest.fixture(scope="class")
    def model(self, dataset: Dataset) -> DRPModel:
        return drevalpy.train(
            "NaiveDrugMeanPredictor", dataset, hyperparameter_tuning=False, quality=CurveQuality.off(), measure="AUC"
        )

    def test_the_model_remembers_its_measure_through_a_checkpoint(self, model: DRPModel, tmp_path: UPath) -> None:
        model.save(UPath(tmp_path) / "model")

        assert model.measure == "AUC"
        assert DRPModel.load(UPath(tmp_path) / "model.zip").measure == "AUC"

    def test_predicting_compares_against_the_model_s_measure(self, model: DRPModel, dataset: Dataset) -> None:
        result = predict_observed(model, dataset)

        assert result.measure == "AUC"
        assert len(result.predictions) == int(np.isfinite(dataset.get_response_layer("AUC")).sum())

    def test_another_measure_set_on_the_dataset_is_rejected(self, model: DRPModel, dataset: Dataset) -> None:
        with pytest.raises(InvalidInputError, match="trained on 'AUC', but the dataset is set to 'LN_IC50'"):
            predict_observed(model, dataset.with_measure("LN_IC50"))

    def test_an_untrained_model_cannot_predict(self, dataset: Dataset) -> None:
        with pytest.raises(InvalidInputError, match="must be trained or loaded"):
            predict_observed(DRPModel(ModelSpec.from_name("NaiveDrugMeanPredictor")), dataset)
