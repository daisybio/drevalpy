"""Tests for MOLIR utility helpers.

The ``_realign_omic_matrix`` behaviour tests live in ``test_omics.py`` since the
helper was split into ``molir/_omics.py`` so the registered predictors could
reach it without importing ``pytorch_lightning``; what remains of it here is the
guard on the compatibility re-export ``utils.py`` still exposes.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.molir import _omics, utils


def test_utils_reexports_the_realign_helper_from_omics() -> None:
    assert utils._realign_omic_matrix is _omics._realign_omic_matrix


def test_generate_triplets_indices_picks_near_and_far_samples() -> None:
    y = np.array([0.0, 0.05, 5.0])

    positive, negative = utils.generate_triplets_indices(y, positive_range=0.1, negative_range=1.0, random_seed=0)

    assert positive.shape == negative.shape == (3,)
    # The two clustered responses are each other's positive; the outlier is their negative.
    assert positive[0] == 1
    assert positive[1] == 0
    assert negative[0] == negative[1] == 2


def test_generate_triplets_indices_falls_back_to_the_sample_itself_for_a_single_response() -> None:
    """A one-element validation split has no other sample to draw a triplet from."""
    positive, negative = utils.generate_triplets_indices(
        np.array([2.0]), positive_range=0.1, negative_range=1.0, random_seed=0
    )

    assert positive.tolist() == [0]
    assert negative.tolist() == [0]


def test_generate_triplets_indices_falls_back_to_the_closest_sample() -> None:
    """No response sits inside the positive range, so the nearest one is used."""
    y = np.array([0.0, 3.0, 10.0])

    positive, _ = utils.generate_triplets_indices(y, positive_range=0.0, negative_range=1.0, random_seed=0)

    assert positive.tolist() == [1, 0, 1]


def _omic_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gene_expression = np.arange(12, dtype=np.float32).reshape(4, 3)
    mutations = np.ones((4, 2), dtype=np.float32)
    copy_number = np.zeros((4, 1), dtype=np.float32)
    return gene_expression, mutations, copy_number


def test_create_dataset_and_loaders_without_validation_returns_only_a_train_loader() -> None:
    gene_expression, mutations, copy_number = _omic_matrices()

    train_loader, val_loader = utils.create_dataset_and_loaders(
        batch_size=2,
        gene_expression=gene_expression,
        mutations=mutations,
        copy_number=copy_number,
        response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        pair_idx=np.array([0, 1, 2, 3]),
    )

    assert val_loader is None
    assert len(train_loader.dataset) == 4
    gex, mut, cnv, response = next(iter(train_loader))
    assert gex.shape == (2, 3)
    assert mut.shape == (2, 2)
    assert cnv.shape == (2, 1)
    assert response.shape == (2, 1)


def test_create_dataset_and_loaders_builds_a_validation_loader() -> None:
    gene_expression, mutations, copy_number = _omic_matrices()

    _, val_loader = utils.create_dataset_and_loaders(
        batch_size=2,
        gene_expression=gene_expression,
        mutations=mutations,
        copy_number=copy_number,
        response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        pair_idx=np.array([0, 1, 2, 3]),
        val_gene_expression=gene_expression,
        val_mutations=mutations,
        val_copy_number=copy_number,
        val_response=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        val_pair_idx=np.array([0, 1, 2]),
    )

    assert val_loader is not None
    # drop_last is False for validation, so the trailing incomplete batch survives.
    assert len(val_loader.dataset) == 3
    assert sum(batch[0].shape[0] for batch in val_loader) == 3


def test_create_dataset_and_loaders_rejects_partial_validation_omics() -> None:
    gene_expression, mutations, copy_number = _omic_matrices()

    with pytest.raises(ValueError, match="val_mutations and val_copy_number are required"):
        utils.create_dataset_and_loaders(
            batch_size=2,
            gene_expression=gene_expression,
            mutations=mutations,
            copy_number=copy_number,
            response=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            pair_idx=np.array([0, 1, 2, 3]),
            val_gene_expression=gene_expression,
            val_response=np.array([5.0, 6.0], dtype=np.float32),
            val_pair_idx=np.array([0, 1]),
        )
