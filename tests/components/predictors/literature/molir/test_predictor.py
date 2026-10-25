"""Tests for the MOLIR predictor, its triplet-loss utilities and the lightning-free omics helpers."""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from drevalpy.components.predictors.literature.molir import _omics, utils
from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.molir.utils import MOLIModel
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors.literature._helpers import (
    assert_corrected_training_result,
    assert_frozen_reference,
    assert_state_round_trip,
    assert_trainable_parameters_changed,
    finite_predictions,
    seed_literature,
    single_drug_omics_batch,
    trainable_parameter_vector,
)


def _loss_model(gamma: float) -> MOLIModel:
    hyperparameters = dict(MOLIRPredictor.get_default_hyperparameters())
    hyperparameters.update(
        {
            "gamma": gamma,
            "mini_batch": 4,
            "h_dim1": 2,
            "h_dim2": 2,
            "h_dim3": 2,
            "dropout_rate": 0.0,
            "epochs": 1,
        }
    )
    model = MOLIModel(hyperparameters, 2, 2, 2)
    model.positive_range = 0.1
    model.negative_range = 0.5
    return model


def test_gamma_weights_triplet_loss_and_its_embedding_gradient() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [0.1, 0.8], [0.9, 0.2]],
        dtype=torch.float32,
    )
    predictions = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float32)
    response = torch.tensor([0.0, 0.05, 1.0, 1.05], dtype=torch.float32)

    unweighted_embeddings = embeddings.clone().requires_grad_()
    unweighted_loss = _loss_model(0.0)._compute_loss(unweighted_embeddings, predictions, response)
    unweighted_loss.backward()
    weighted_embeddings = embeddings.clone().requires_grad_()
    weighted_loss = _loss_model(1.0)._compute_loss(weighted_embeddings, predictions, response)
    weighted_loss.backward()

    assert weighted_loss > unweighted_loss
    torch.testing.assert_close(unweighted_embeddings.grad, torch.zeros_like(unweighted_embeddings))
    assert torch.count_nonzero(weighted_embeddings.grad).item() > 0


def test_molir_predictor_registry_name() -> None:
    assert get_predictor("molir") is MOLIRPredictor


class TestTinyMOLIRFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state_without_feature_names(self, seed: int, tmp_path) -> None:
        hyperparameters = {
            "epochs": 1,
            # One full batch: BatchNorm over two rows amplifies platform rounding to ~1e-2.
            "mini_batch": 6,
            "h_dim1": 4,
            "h_dim2": 4,
            "h_dim3": 4,
            "dropout_rate": 0.0,
        }
        batch = single_drug_omics_batch(tmp_path)
        seed_literature(seed)
        pristine = MOLIRPredictor(hyperparameters)._build_model((6, 5, 4))
        initial = trainable_parameter_vector(pristine)

        seed_literature(seed)
        predictor = MOLIRPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        fitted = predictor._models["d1"]
        assert_trainable_parameters_changed(initial, fitted)
        assert fitted.checkpoint_callback is not None
        expected = finite_predictions(predictor, batch)
        # The gamma correction supersedes the v2 baseline in training.json for MOLIR.
        assert_corrected_training_result(
            "molir_gamma_weighting",
            seed,
            expected,
            fitted.checkpoint_callback.best_model_score,
        )
        assert_frozen_reference("molir", seed)

        assert_state_round_trip(predictor, batch, expected)


class TestMOLIRTrainer:
    pytestmark = pytest.mark.slow

    def test_runs_on_the_cpu_without_a_default_logger(self, tmp_path, monkeypatch) -> None:
        real_trainer = pl.Trainer
        seen: list[dict[str, object]] = []

        def spy(*args, **kwargs):
            seen.append(dict(kwargs))
            return real_trainer(*args, **kwargs)

        monkeypatch.setattr(pl, "Trainer", spy)
        monkeypatch.chdir(tmp_path)
        hyperparameters = {"epochs": 1, "mini_batch": 6, "h_dim1": 2, "h_dim2": 2, "h_dim3": 2}

        MOLIRPredictor(hyperparameters).fit(single_drug_omics_batch(tmp_path / "ckpt"))

        (kwargs,) = seen
        assert kwargs["accelerator"] == "cpu"
        assert kwargs["logger"] is False
        assert not (tmp_path / "lightning_logs").exists()


def test_utils_reexports_the_realign_helper_from_omics() -> None:
    assert utils._realign_omic_matrix is _omics._realign_omic_matrix


def test_generate_triplets_indices_picks_near_and_far_pairs() -> None:
    y = np.array([0.0, 0.05, 5.0])

    positive, negative = utils.generate_triplets_indices(y, positive_range=0.1, negative_range=1.0, random_seed=0)

    assert positive.shape == negative.shape == (3,)
    # The two clustered responses are each other's positive; the outlier is their negative.
    assert positive[0] == 1
    assert positive[1] == 0
    assert negative[0] == negative[1] == 2


def test_generate_triplets_indices_falls_back_for_a_single_response_and_to_the_closest_pair() -> None:
    positive, negative = utils.generate_triplets_indices(
        np.array([2.0]), positive_range=0.1, negative_range=1.0, random_seed=0
    )
    assert (positive.tolist(), negative.tolist()) == ([0], [0])

    y = np.array([0.0, 3.0, 10.0])
    positive, _ = utils.generate_triplets_indices(y, positive_range=0.0, negative_range=1.0, random_seed=0)
    assert positive.tolist() == [1, 0, 1]


def test_generate_triplets_indices_uses_the_process_numpy_seed_by_default() -> None:
    y = np.array([0.0, 0.02, 0.04, 1.0, 1.02, 1.04])
    np.random.seed(123)
    state_before = np.random.get_state()

    first = utils.generate_triplets_indices(y, positive_range=0.05, negative_range=0.5)
    state_after = np.random.get_state()
    np.random.seed(123)
    second = utils.generate_triplets_indices(y, positive_range=0.05, negative_range=0.5)

    assert state_before[2] != state_after[2]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_generate_triplets_indices_explicit_seed_is_independent_of_process_numpy_seed() -> None:
    y = np.array([0.0, 0.02, 0.04, 1.0, 1.02, 1.04])
    np.random.seed(1)
    first = utils.generate_triplets_indices(y, positive_range=0.05, negative_range=0.5, random_seed=77)
    np.random.seed(999)
    second = utils.generate_triplets_indices(y, positive_range=0.05, negative_range=0.5, random_seed=77)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize(
    ("values", "model_features", "incoming_features", "expected"),
    [
        ([1.0, 2.0, 3.0], ["g1", "g2"], ["g0", "g1", "g2"], [2.0, 3.0]),
        ([6.0, 7.0, 8.0], ["c2", "c1"], ["c0", "c1", "c2"], [8.0, 7.0]),
        ([10.0, 20.0], ["a", "b", "c"], ["b", "d"], [0.0, 10.0, 0.0]),
        ([10.0, 20.0, 30.0], ["a", "b", "c"], ["c", "a", "b"], [20.0, 30.0, 10.0]),
        ([10.0, 20.0, 30.0], ["a", "b", "missing"], ["c", "a", "b"], [20.0, 30.0, 0.0]),
    ],
    ids=["narrow", "reorder", "zero-fill", "equal-width-permutation", "equal-width-missing"],
)
def test_realign_omic_matrix(values, model_features, incoming_features, expected) -> None:
    result = _realign_omic_matrix(np.array([values]), model_features, incoming_features)

    np.testing.assert_allclose(result, [expected])


def test_realign_omic_matrix_passes_matching_features_through_unchanged() -> None:
    values = np.array([[1.0, 2.0]])

    assert _realign_omic_matrix(values, ["a", "b"], ["a", "b"]) is values
