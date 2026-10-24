"""Tests for the shared three-omic loader construction."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, make_omics_loaders

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


def test_each_batch_carries_the_three_views_and_a_response_column() -> None:
    split = _split(n_pairs=4)
    train_loader, val_loader = make_omics_loaders(split, None, batch_size=4)

    assert val_loader is None
    expression, mutations, copy_number, response = next(iter(train_loader))
    assert (expression.shape, mutations.shape, copy_number.shape) == ((4, EXPR_DIM), (4, MUT_DIM), (4, CNV_DIM))
    assert response.shape == (4, 1)
    np.testing.assert_allclose(expression.numpy(), split.gene_expression[split.pair_idx])


def test_training_drops_the_trailing_incomplete_batch_but_validation_keeps_it() -> None:
    train_loader, val_loader = make_omics_loaders(_split(n_pairs=5), _split(n_pairs=5), batch_size=2)

    assert sum(batch[0].shape[0] for batch in train_loader) == 4
    assert val_loader is not None
    assert sum(batch[0].shape[0] for batch in val_loader) == 5


def test_an_already_two_dimensional_response_is_passed_through() -> None:
    train_loader, _ = make_omics_loaders(_split(two_dimensional_response=True), None, batch_size=2)

    assert next(iter(train_loader))[3].shape == (2, 1)


def test_neither_loader_shuffles_and_the_split_is_frozen() -> None:
    split = _split(n_pairs=4)
    train_loader, _ = make_omics_loaders(split, None, batch_size=1)

    assert [float(batch[3].item()) for batch in train_loader] == pytest.approx(split.response.tolist())
    with pytest.raises(AttributeError):
        split.pair_idx = np.zeros(1, dtype=np.int64)  # type: ignore[misc]
