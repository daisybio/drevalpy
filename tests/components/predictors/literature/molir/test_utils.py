"""Tests for MOLIR utility helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.molir import _omics, utils


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
