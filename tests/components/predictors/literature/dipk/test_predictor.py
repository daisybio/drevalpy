"""Tests for the DIPK block predictor.

The training loop itself needs the network-gated BIONIC and MolGNet artifacts, so
what is exercised here is everything around it: sample construction, the collate
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
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block, ragged_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.models.synthetic_fixtures import multi_drug_response


def _dipk_batch() -> ModelInputBatch:
    """Build a batch with the three blocks DIPK consumes.

    :returns: Featurized ``ModelInputBatch`` with two cell lines and two drugs.
    """
    molgnet = np.empty(2, dtype=object)
    molgnet[:] = [
        np.arange(6, dtype=np.float32).reshape(3, 2),
        np.arange(4, dtype=np.float32).reshape(2, 2),
    ]
    return ModelInputBatch.from_response(
        multi_drug_response(),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        cell_line_blocks={
            "gene_expression": numeric_feature_block(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)),
            "bionic_features": numeric_feature_block(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        },
        drug_blocks={"molgnet_features": ragged_feature_block(molgnet)},
        training_context=TrainingContext(checkpoint_dir=Path(tempfile.mkdtemp())),
    )


def test_dipk_predictor_registry_name() -> None:
    ensure_predictor_registered("dipk")
    assert get_predictor("dipk") is DIPKPredictor


def test_default_hyperparameters_cover_the_training_loop_knobs() -> None:
    defaults = DIPKPredictor.get_default_hyperparameters()

    assert {"batch_size", "lr", "epochs", "patience"} <= set(defaults)


class TestBuildSamples:
    def test_one_sample_per_pair_with_the_three_blocks(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        samples = predictor._build_samples(batch.cell_line_pair_idx, batch.drug_pair_idx, batch)

        assert len(samples) == 4
        assert set(samples[0]) == {"molgnet_features", "gene_expression", "bionic_features"}

    def test_pair_indices_select_the_entity_rows(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        samples = predictor._build_samples(batch.cell_line_pair_idx, batch.drug_pair_idx, batch)

        # Pair 2 is (cell line 1, drug 0).
        torch.testing.assert_close(samples[2]["gene_expression"], torch.tensor([0.3, 0.4]))
        assert samples[2]["molgnet_features"].shape == (3, 2)

    def test_a_response_adds_the_target_entry(self) -> None:
        predictor = DIPKPredictor()
        batch = _dipk_batch()

        samples = predictor._build_samples(
            batch.cell_line_pair_idx,
            batch.drug_pair_idx,
            batch,
            response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

        assert samples[3]["ic50"].tolist() == [4.0]


class TestCollateFn:
    def _samples(self) -> list[dict[str, torch.Tensor]]:
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
        collated = _CollateFn(train=False)(self._samples())

        assert collated["molgnet_features"].shape == (2, 3, 2)
        assert collated["molgnet_features"][1, 1:].abs().sum().item() == 0.0

    def test_the_mask_marks_only_the_real_atoms(self) -> None:
        collated = _CollateFn(train=False)(self._samples())

        assert collated["molgnet_mask"].dtype is torch.bool
        assert collated["molgnet_mask"][1].tolist() == [True, False, False]

    def test_targets_are_included_only_in_training_mode(self) -> None:
        train = _CollateFn(train=True)(self._samples())
        predict = _CollateFn(train=False)(self._samples())

        assert train["ic50_values"].flatten().tolist() == [5.0, 6.0]
        assert "ic50_values" not in predict


class TestDIPKDataset:
    def test_length_is_the_sample_count(self) -> None:
        assert len(_DIPKDataset([{"a": torch.zeros(1)}, {"a": torch.ones(1)}])) == 2

    def test_indexing_returns_the_stored_sample(self) -> None:
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

    def test_set_state_requires_hyperparameters(self) -> None:
        with pytest.raises(PredictorStateError, match="hyperparameters dict"):
            DIPKPredictor().set_state({"payload": b"x"})

    def test_set_state_requires_payload_bytes(self) -> None:
        with pytest.raises(PredictorStateError, match="payload bytes"):
            DIPKPredictor().set_state({"hyperparameters": {}, "payload": "not-bytes"})
