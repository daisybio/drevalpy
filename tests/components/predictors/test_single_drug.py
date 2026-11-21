"""Tests for the single-drug predictor base and the per-drug sklearn predictors built on it."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import pytest
from joblib.externals import cloudpickle
from upath import UPath

from drevalpy._utils.seed import seed_everything
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.abstract.single_drug import SingleDrugPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.sklearn_models import SingleDrugElasticNetPredictor, SingleDrugRandomForestPredictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors._helpers import pickle_round_trip


@dataclass
class _Fitted:
    draws: tuple[int, float, float]
    threads: int
    pid: int
    batch: ModelInputBatch


class _RecordingPredictor(SingleDrugPredictor):
    """Return what each drug's fit saw; skip drug ``skip`` and fail on drug ``boom``."""

    sample_inputs: ClassVar[Inputs] = Inputs.matrix()

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {}

    def _fit_drug(self, drug_id: str, batch: ModelInputBatch) -> _Fitted | None:
        import torch

        if drug_id == "skip":
            return None
        if drug_id == "boom":
            raise RuntimeError("boom does not converge")
        draws = (random.getrandbits(24), float(np.random.rand()), float(torch.rand(1)))
        return _Fitted(draws, torch.get_num_threads(), os.getpid(), batch)

    def _predict_drug(self, drug_id: str, model: _Fitted, batch: ModelInputBatch) -> np.ndarray:
        return np.full(batch.n_pairs, float(np.mean(model.batch.response)))


def _batch(
    pairs: dict[str, list[int]],
    *,
    n_samples: int = 3,
    n_features: int = 2,
    n_jobs: int = 1,
    checkpoint_dir: str = "/base",
    early_stopping_response: ResponseBatch | None = None,
) -> ModelInputBatch:
    sample_ids = np.array([f"s{i}" for i in range(n_samples)])
    drug_ids = np.array([drug for drug, samples in pairs.items() for _ in samples])
    response = ResponseBatch(
        response=np.arange(len(drug_ids), dtype=np.float64),
        sample_ids=sample_ids[[sample for samples in pairs.values() for sample in samples]],
        drug_ids=drug_ids,
    )
    values = np.arange(n_samples * n_features, dtype=np.float64).reshape(n_samples, n_features)
    sample = FeaturizerResult.from_blocks({"gene_expression": numeric_feature_block(values)}, sample_ids)
    return ModelInputBatch.from_response(
        response,
        sample=sample,
        early_stopping_response=early_stopping_response,
        training_context=TrainingContext(checkpoint_dir=UPath(checkpoint_dir), n_jobs=n_jobs),
    )


def _fit(batch: ModelInputBatch) -> dict[str, _Fitted]:
    predictor = _RecordingPredictor()
    predictor.fit(batch)
    return predictor._models


@pytest.fixture(autouse=True)
def _restore_global_rngs():
    import torch

    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def test_each_drug_gets_its_own_pairs_checkpoint_dir_and_a_single_process() -> None:
    models = _fit(_batch({"d2": [0, 1], "d1": [1, 2], "a/b c:*?": [0]}))

    assert list(models) == sorted(["d2", "d1", "a/b c:*?"])
    for drug_id, model in models.items():
        assert model.batch.drug_ids.tolist() == [drug_id] * model.batch.n_pairs
        assert model.batch.training_context.n_jobs == 1
    np.testing.assert_array_equal(models["d1"].batch.sample_ids, ["s1", "s2"])
    directories = [UPath(model.batch.training_context.checkpoint_dir) for model in models.values()]
    assert len(set(directories)) == 3
    assert all(d.parent == UPath("/base") and d.name.startswith("drug_") for d in directories)
    again = _fit(_batch({"d1": [0]}))["d1"].batch.training_context.checkpoint_dir
    assert again == models["d1"].batch.training_context.checkpoint_dir


def test_a_drug_batch_keeps_only_the_rows_its_pairs_and_stopping_pairs_use() -> None:
    stopping = ResponseBatch(np.array([5.0, 6.0]), np.array(["s3", "s4"]), np.array(["d1", "d2"]))
    models = _fit(_batch({"d1": [2, 0, 2], "d2": [1]}, n_samples=5, early_stopping_response=stopping))

    d1 = models["d1"].batch
    np.testing.assert_array_equal(d1.sample.entity_ids, ["s0", "s2", "s3"])
    np.testing.assert_array_equal(d1.sample.entity_ids[d1.sample_pair_idx], d1.sample_ids)
    np.testing.assert_array_equal(d1.sample.matrix[d1.sample_pair_idx], [[4.0, 5.0], [0.0, 1.0], [4.0, 5.0]])
    assert d1.early_stopping_response is not None
    np.testing.assert_array_equal(d1.early_stopping_response.sample_ids, ["s3"])


@pytest.mark.parametrize("n_jobs", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_response_layers_stay_aligned_through_the_nan_filter_and_the_per_drug_cut(n_jobs: int) -> None:
    stopping = ResponseBatch(
        np.array([5.0, 6.0]), np.array(["s0", "s1"]), np.array(["d1", "d2"]), {"regulation": np.array([1.0, 0.0])}
    )
    batch = _batch({"d1": [2, 0, 1], "d2": [1, 2]}, n_jobs=n_jobs, early_stopping_response=stopping)
    batch = replace(
        batch,
        response=np.array([0.0, np.nan, 2.0, 3.0, 4.0]),
        response_layers={"regulation": np.array([-1.0, 1.0, 0.0, np.nan, -1.0])},
    )

    models = _fit(batch)

    d1, d2 = models["d1"].batch, models["d2"].batch
    np.testing.assert_array_equal(d1.sample_ids, ["s2", "s1"])
    np.testing.assert_array_equal(d1.response_layers["regulation"], [-1.0, 0.0])
    np.testing.assert_array_equal(d2.response_layers["regulation"], [np.nan, -1.0])
    assert d1.early_stopping_response is not None
    np.testing.assert_array_equal(d1.early_stopping_response.layers["regulation"], [1.0])


def test_a_pickled_drug_batch_is_much_smaller_than_the_full_batch() -> None:
    batch = _batch({"d1": [0, 1], "d2": [2, 3]}, n_samples=2000, n_features=50)

    drug_batch = _fit(batch)["d1"].batch

    # loky sends a task to its worker with cloudpickle.
    assert len(cloudpickle.dumps(("d1", drug_batch))) * 50 < len(cloudpickle.dumps(batch))


def test_pairs_are_scored_by_their_drugs_model_and_drugs_without_one_get_nan() -> None:
    predictor = _RecordingPredictor()
    predictor.fit(_batch({"d1": [0, 1], "skip": [2], "d2": [0, 2]}))
    assert set(predictor._models) == {"d1", "d2"}

    query = ResponseBatch(None, np.array(["s0", "s1", "s2", "s0"]), np.array(["d2", "skip", "d1", "unseen"]))
    sample = _batch({"d1": [0]}).sample
    predictions = predictor.predict(ModelInputBatch.from_response(query, sample=sample))

    np.testing.assert_array_equal(predictions[[0, 2]], [3.5, 0.5])
    assert np.isnan(predictions[[1, 3]]).all()


class TestFanOut:
    pytestmark = pytest.mark.slow

    def test_n_jobs_changes_neither_the_draws_nor_the_final_rng_state(self) -> None:
        results, after = {}, {}
        for n_jobs in (1, 2):
            seed_everything(11)
            models = _fit(_batch({"d1": [0], "d2": [1], "d3": [2]}, n_jobs=n_jobs))
            results[n_jobs] = {drug_id: model.draws for drug_id, model in models.items()}
            after[n_jobs] = random.getrandbits(24), float(np.random.rand())

        assert results[1] == results[2]
        assert len(set(results[1].values())) == 3
        assert after[1] == after[2]

    def test_workers_are_single_threaded_processes(self) -> None:
        workers = _fit(_batch({"d1": [0], "d2": [1]}, n_jobs=2))
        assert {model.pid for model in workers.values()}.isdisjoint({os.getpid()})
        assert {model.threads for model in workers.values()} == {1}
        serial = _fit(_batch({"d1": [0], "d2": [1]}))
        assert {model.pid for model in serial.values()} == {os.getpid()}

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_an_error_for_one_drug_is_raised_for_all(self, n_jobs: int) -> None:
        with pytest.raises(RuntimeError, match="boom does not converge"):
            _fit(_batch({"d1": [0], "boom": [1], "d3": [2]}, n_jobs=n_jobs))

    def test_an_unseeded_forest_predicts_the_same_for_one_and_two_jobs(self) -> None:
        batch = _batch({"d1": [0, 1, 2, 3], "d2": [3, 2, 1, 0]}, n_samples=4, n_features=3)
        predictions = []
        for n_jobs in (1, 2):
            seed_everything(3)
            predictor = SingleDrugRandomForestPredictor(
                {"n_estimators": 5, "max_samples": 0.5, "random_state": None, "n_jobs": -1}
            )
            predictor.fit(replace(batch, training_context=TrainingContext(n_jobs=n_jobs)))
            predictions.append(predictor.predict(batch))
        np.testing.assert_array_equal(predictions[0], predictions[1])


def test_a_lone_drug_runs_in_the_calling_process_with_the_rng_state_it_finds() -> None:
    seed_everything(5)
    expected = random.getrandbits(24)
    seed_everything(5)
    (model,) = _fit(_batch({"d1": [0]}, n_jobs=2)).values()
    assert model.draws[0] == expected
    assert model.pid == os.getpid()


def test_a_drug_depends_on_the_global_seed_but_not_on_the_other_drugs() -> None:
    seed_everything(11)
    alone = _fit(_batch({"d1": [0], "d2": [1]}))
    seed_everything(11)
    with_d3 = _fit(_batch({"d3": [2], "d2": [1], "d1": [0]}))
    seed_everything(12)
    reseeded = _fit(_batch({"d1": [0], "d2": [1]}))

    assert alone["d2"].draws == with_d3["d2"].draws
    assert alone["d1"].draws != reseeded["d1"].draws


def _fitted_elastic_net(*, n_jobs: int = 1) -> SingleDrugElasticNetPredictor:
    predictor = SingleDrugElasticNetPredictor({"alpha": 0.1, "l1_ratio": 0.5})
    predictor.fit(_batch({"d1": [0, 1, 2], "d2": [0, 1, 2]}, n_features=3, n_jobs=n_jobs))
    return predictor


def test_elastic_net_fits_one_estimator_per_drug_on_sample_features_only() -> None:
    predictor = _fitted_elastic_net()

    assert SingleDrugElasticNetPredictor.drug_inputs.kind == "none"
    assert set(predictor._models) == {"d1", "d2"}
    assert {estimator.n_features_in_ for estimator in predictor._models.values()} == {3}
    query = ResponseBatch(None, np.array(["s0", "s0", "s0"]), np.array(["d1", "d2", "d3"]))
    predictions = predictor.predict(
        ModelInputBatch.from_response(query, sample=_batch({"d1": [0]}, n_features=3).sample)
    )
    assert predictions[0] < predictions[1]
    assert np.isnan(predictions[2])


def test_pickle_round_trips_every_per_drug_estimator() -> None:
    predictor = _fitted_elastic_net()
    assert SingleDrugElasticNetPredictor().is_fitted() is False

    restored = pickle_round_trip(predictor)

    assert restored.is_fitted() is True
    assert set(restored._models) == {"d1", "d2"}
    assert restored._hyperparameters["alpha"] == pytest.approx(0.1)
