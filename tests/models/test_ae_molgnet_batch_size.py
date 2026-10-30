"""Tests for the autoencoder batch size of RandomForestAEMolGNet."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.baselines import ae_molgnet_random_forest as ae_module
from drevalpy.models.baselines.ae_molgnet_random_forest import RandomForestAEMolGNet
from drevalpy.models.DIPK.gene_expression_encoder import GeneExpressionEncoder

_N_GENES = 6
_BATCH = 32


def _model() -> RandomForestAEMolGNet:
    """
    Build the model with its shipped hyperparameters.

    :returns: model with the first shipped hyperparameter set
    """
    model = RandomForestAEMolGNet()
    model.build_model(model.get_hyperparameter_set()[0])
    return model


def _cell_lines(n_cell_lines: int) -> tuple[FeatureDataset, np.ndarray]:
    """
    Build a toy gene expression feature set.

    :param n_cell_lines: number of cell lines
    :returns: the feature set and the cell line identifiers
    """
    ids = np.array([f"CL{i}" for i in range(n_cell_lines)])
    rng = np.random.default_rng(0)
    features = {cl: {"gene_expression": rng.normal(size=_N_GENES)} for cl in ids}
    meta = {"gene_expression": [f"G{i}" for i in range(_N_GENES)]}
    return FeatureDataset(features=features, meta_info=meta), ids


def test_shipped_batch_size_is_32_and_dipk_keeps_1024() -> None:
    """The model ships batch 32 for its autoencoder, the DIPK default stays 1024."""
    assert RandomForestAEMolGNet().get_hyperparameter_set()[0]["batch_size_autoencoder"] == _BATCH
    assert _model()._ae_batch_size == _BATCH
    dipk_yaml = Path(ae_module.__file__).parents[1] / "DIPK" / "hyperparameters.yaml"
    with open(dipk_yaml) as handle:
        assert yaml.safe_load(handle)["DIPK"]["batch_size_autoencoder"] == [1024]


def test_code_default_without_the_key_is_32() -> None:
    """A configuration without batch_size_autoencoder gets 32 as well, not the former 1024."""
    model = RandomForestAEMolGNet()
    hpams = dict(model.get_hyperparameter_set()[0])
    del hpams["batch_size_autoencoder"]
    model.build_model(hpams)
    assert model._ae_batch_size == _BATCH


@pytest.mark.parametrize(
    "n_train, expected_train, expected_val",
    [(2 * _BATCH + 1, 2 * _BATCH, 5), (2 * _BATCH, 2 * _BATCH, 4)],
)
def test_trailing_batch_of_one_row_moves_to_validation(
    monkeypatch: pytest.MonkeyPatch, n_train: int, expected_train: int, expected_val: int
) -> None:
    """
    With batch 32 a training set of 32k + 1 rows leaves a trailing batch of one row, which BatchNorm cannot train on.

    One row moves to the validation set; other sizes stay untouched.

    :param monkeypatch: pytest fixture
    :param n_train: number of training cell lines
    :param expected_train: expected rows in the training matrix handed to the autoencoder
    :param expected_val: expected rows in the validation matrix handed to the autoencoder
    """
    monkeypatch.delenv("DREVAL_AE_CACHE", raising=False)
    seen: dict = {}
    encoder = MagicMock(spec=GeneExpressionEncoder)  # a real type: typeguard checks the return value

    def _fake_training(train, validation, **kwargs):
        seen.update(train=len(train), validation=len(validation), batch_size=kwargs["batch_size"])
        return encoder

    monkeypatch.setattr(ae_module, "train_gene_expession_autoencoder", _fake_training)
    model = _model()
    features, ids = _cell_lines(n_train + 4)
    assert model._fit_or_load_encoder(ids[:n_train], features, ids[n_train:]) is encoder
    assert seen == {"train": expected_train, "validation": expected_val, "batch_size": _BATCH}
