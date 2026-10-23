"""Tests for the DIPK block predictor.

The training loop itself needs the network-gated BIONIC and MolGNet artifacts, so
what is exercised here is everything around it: per-pair dict construction, the collate
function, the list-backed dataset, and the state/guard paths. Those are also the
parts that carry the ``torch`` imports this module defers (see
``tests/test_import_cost_policy.py``), so they are worth pinning directly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.dipk.predictor import (
    DIPKPredictor,
    _CollateFn,
    _DIPKDataset,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block, ragged_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    seed_literature,
    trainable_parameter_vector,
    two_by_two_batch,
)


def _dipk_batch() -> ModelInputBatch:
    """Build a batch with the three blocks DIPK consumes.

    :returns: Featurized ``ModelInputBatch`` with two samples and two drugs.
    """
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.arange(4, dtype=np.float32).reshape(2, 2),
    ]
    return two_by_two_batch(
        sample_blocks={
            "gene_expression": numeric_feature_block(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
            "bionic_features": numeric_feature_block(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        },
        drug_blocks={"molgnet_features": ragged_feature_block(molgnet)},
        checkpoint_dir=Path(tempfile.mkdtemp()),
    )


def _trainable_dipk_batch(tmp_path: Path) -> ModelInputBatch:
    """Build full-width inputs with a validation split disjoint from training."""
    rng = np.random.default_rng(321)
    cell_ids = np.array(["train-1", "train-2", "val-1", "val-2"])
    drug_ids = np.array(["d1", "d2"])
    response = ResponseBatch(
        response=np.array([-1.0, 0.5, 1.5, 3.0], dtype=np.float32),
        sample_ids=np.array(["train-1", "train-1", "train-2", "train-2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    validation = ResponseBatch(
        response=np.array([0.25, 2.25], dtype=np.float32),
        sample_ids=np.array(["val-1", "val-2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [rng.normal(size=(2, 768)).astype(np.float32), rng.normal(size=(3, 768)).astype(np.float32)]
    return ModelInputBatch.from_response(
        response,
        sample=FeaturizerResult.from_blocks(
            {
                "gene_expression": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
                "bionic_features": numeric_feature_block(rng.normal(size=(4, 512)).astype(np.float32)),
            },
            cell_ids,
        ),
        drug=FeaturizerResult.from_blocks({"molgnet_features": ragged_feature_block(molgnet)}, drug_ids),
        sample_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=validation,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )


def test_dipk_predictor_registry_name() -> None:
    assert get_predictor("dipk") is DIPKPredictor


def test_default_hyperparameters_cover_the_training_loop_knobs() -> None:
    defaults = DIPKPredictor.get_default_hyperparameters()

    assert {"batch_size", "lr", "epochs", "patience"} <= set(defaults)


class TestFrozenDIPKNumericalReference:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_v2(self, seed: int) -> None:
        assert_frozen_reference("dipk", seed)


class TestTinyDIPKFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int, tmp_path: Path) -> None:
        hyperparameters = {
            "batch_size": 4,
            "lr": 0.001,
            "heads": 1,
            "fc_layer_num": 3,
            "fc_layer_dim": [8, 8, 8, 8, 8, 8],
            "dropout_rate": 0.0,
            "epochs": 1,
            "patience": 1,
        }
        seed_literature(seed)
        initial_predictor = DIPKPredictor(hyperparameters)
        initial = trainable_parameter_vector(initial_predictor._build_model())

        seed_literature(seed)
        predictor = DIPKPredictor(hyperparameters)
        batch = _trainable_dipk_batch(tmp_path)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)

        predictions = predictor.predict(batch)
        assert np.isfinite(predictions).all()
        assert np.isfinite(np.mean((predictions - batch.response) ** 2))
        assert_training_predictions("dipk", seed, predictions)

        restored = DIPKPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(restored.predict(batch), predictions, rtol=1e-6, atol=1e-7)


class TestBuildPairs:
    def test_one_dict_per_pair_with_the_three_blocks(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        pairs = predictor._build_pairs(batch.sample_pair_idx, batch.drug_pair_idx, batch)

        assert len(pairs) == 4
        assert set(pairs[0]) == {"molgnet_features", "gene_expression", "bionic_features"}

    def test_pair_indices_select_the_entity_rows(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        pairs = predictor._build_pairs(batch.sample_pair_idx, batch.drug_pair_idx, batch)

        # Pair 2 is (sample 1, drug 0).
        torch.testing.assert_close(pairs[2]["gene_expression"], torch.tensor([0.3, 0.4]))
        assert pairs[2]["molgnet_features"].shape == (3, 2)

    def test_a_response_adds_the_target_entry(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        pairs = predictor._build_pairs(
            batch.sample_pair_idx,
            batch.drug_pair_idx,
            batch,
            response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

        assert pairs[3]["ic50"].tolist() == [4.0]


class TestCollateFn:
    def _pairs(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "molgnet_features": torch.ones(3, 2),
                "gene_expression": torch.tensor([0.1, 0.2]),
                "bionic_features": torch.tensor([1.0, 2.0]),
                "ic50": torch.tensor([5.0]),
            },
            {
                "molgnet_features": torch.ones(1, 2),
                "gene_expression": torch.tensor([0.3, 0.4]),
                "bionic_features": torch.tensor([3.0, 4.0]),
                "ic50": torch.tensor([6.0]),
            },
        ]

    def test_ragged_atoms_are_zero_padded_to_the_longest_molecule(self) -> None:
        collated = _CollateFn(train=False)(self._pairs())

        assert collated["molgnet_features"].shape == (2, 3, 2)
        assert collated["molgnet_features"][1, 1:].abs().sum().item() == 0.0

    def test_the_mask_marks_only_the_real_atoms(self) -> None:
        collated = _CollateFn(train=False)(self._pairs())

        assert collated["molgnet_mask"].dtype is torch.bool
        assert collated["molgnet_mask"][1].tolist() == [True, False, False]

    def test_targets_are_included_only_in_training_mode(self) -> None:
        train = _CollateFn(train=True)(self._pairs())
        predict = _CollateFn(train=False)(self._pairs())

        assert train["ic50_values"].flatten().tolist() == [5.0, 6.0]
        assert "ic50_values" not in predict


class TestDIPKDataset:
    def test_length_is_the_pair_count(self) -> None:
        assert len(_DIPKDataset([{"a": torch.zeros(1)}, {"a": torch.ones(1)}])) == 2

    def test_indexing_returns_the_stored_pair(self) -> None:
        second = {"a": torch.ones(1)}

        assert _DIPKDataset([{"a": torch.zeros(1)}, second])[1] is second

    def test_it_drives_a_dataloader_without_subclassing_torch_dataset(self) -> None:
        """The class deliberately has no ``torch.utils.data.Dataset`` base."""
        from torch.utils.data import DataLoader

        dataset = _DIPKDataset([{"a": torch.zeros(1)}, {"a": torch.ones(1)}])

        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=list)

        assert len(next(iter(loader))) == 2


class TestGuards:
    def test_predict_before_fit_returns_all_nan(self) -> None:
        predictions = DIPKPredictor()._predict(_dipk_batch())

        assert np.isnan(predictions).all()
        assert len(predictions) == 4

    def test_fit_without_early_stopping_data_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="early stopping data"):
            DIPKPredictor()._fit(_dipk_batch())

    def test_is_fitted_is_false_and_state_is_empty_before_training(self) -> None:
        predictor = DIPKPredictor()

        assert predictor.is_fitted() is False
        assert predictor.get_state() == {}

    def test_set_state_requires_a_payload_blob(self) -> None:
        with pytest.raises(PredictorStateError, match="payload byte blob"):
            DIPKPredictor().set_state({"payload": "not-bytes"})

    def test_set_state_requires_the_model_state(self) -> None:
        payload = save_trusted_mapping({"predictor_hyperparameters": DIPKPredictor.get_default_hyperparameters()})

        with pytest.raises(PredictorStateError, match="missing model_state"):
            DIPKPredictor().set_state({"payload": payload})
