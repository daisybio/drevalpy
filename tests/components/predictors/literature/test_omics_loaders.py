"""Tests for the shared three-omic loader construction.

``literature/_omics_loaders.py`` is what MOLIR's ``MOLIModel.fit`` and SuperFELTR's
``train_superfeltr_model`` both call; the two used to carry byte-identical copies of
this code. The contract worth pinning is the asymmetry between the two loaders -
``drop_last=True`` for training, ``False`` for validation - and the response column
reshape, because a regression in either silently changes what the models see.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders

#: ``make_pair_loader`` pulls in torch, which costs more than the fast tier's budget.
pytestmark = pytest.mark.slow

N_ENTITIES = 4
EXPR_DIM = 3
MUT_DIM = 2
CNV_DIM = 1


def _split(n_pairs: int = 5, *, two_dimensional_response: bool = False) -> OmicsSplit:
    response = np.linspace(0.0, 1.0, n_pairs, dtype=np.float32)
    return OmicsSplit(
        gene_expression=np.arange(N_ENTITIES * EXPR_DIM, dtype=np.float32).reshape(N_ENTITIES, EXPR_DIM),
        mutations=np.ones((N_ENTITIES, MUT_DIM), dtype=np.float32),
        copy_number=np.zeros((N_ENTITIES, CNV_DIM), dtype=np.float32),
        response=response.reshape(-1, 1) if two_dimensional_response else response,
        pair_idx=np.arange(n_pairs, dtype=np.int64) % N_ENTITIES,
    )


def test_without_validation_only_a_train_loader_is_built() -> None:
    train_loader, val_loader = make_omics_loaders(_split(), None, batch_size=2)

    assert val_loader is None
    assert len(train_loader.dataset) == 5


def test_each_batch_carries_the_three_views_and_a_response_column() -> None:
    train_loader, _ = make_omics_loaders(_split(), None, batch_size=2)

    expression, mutations, copy_number, response = next(iter(train_loader))

    assert expression.shape == (2, EXPR_DIM)
    assert mutations.shape == (2, MUT_DIM)
    assert copy_number.shape == (2, CNV_DIM)
    assert response.shape == (2, 1)


def test_all_three_views_are_indexed_by_the_same_pair_index() -> None:
    split = _split(n_pairs=4)
    train_loader, _ = make_omics_loaders(split, None, batch_size=4)

    expression, _, _, _ = next(iter(train_loader))

    np.testing.assert_allclose(expression.numpy(), split.gene_expression[split.pair_idx])


def test_training_drops_the_trailing_incomplete_batch() -> None:
    train_loader, _ = make_omics_loaders(_split(n_pairs=5), None, batch_size=2)

    assert sum(batch[0].shape[0] for batch in train_loader) == 4


def test_validation_keeps_the_trailing_incomplete_batch() -> None:
    _, val_loader = make_omics_loaders(_split(), _split(n_pairs=5), batch_size=2)

    assert val_loader is not None
    assert sum(batch[0].shape[0] for batch in val_loader) == 5


def test_an_already_two_dimensional_response_is_passed_through() -> None:
    train_loader, _ = make_omics_loaders(_split(two_dimensional_response=True), None, batch_size=2)

    _, _, _, response = next(iter(train_loader))

    assert response.shape == (2, 1)


def test_neither_loader_shuffles_so_pair_order_is_reproducible() -> None:
    split = _split(n_pairs=4)
    train_loader, _ = make_omics_loaders(split, None, batch_size=1)

    seen = [float(batch[3].item()) for batch in train_loader]

    assert seen == pytest.approx(split.response.tolist())


def test_the_split_is_frozen_so_a_loader_cannot_be_repointed() -> None:
    """``OmicsSplit`` replaced five separately-optional validation arguments."""
    split = _split()

    with pytest.raises(AttributeError):
        split.pair_idx = np.zeros(1, dtype=np.int64)  # type: ignore[misc]
