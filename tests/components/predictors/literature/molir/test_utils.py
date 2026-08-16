"""Tests for MOLIR utility helpers.

The ``_realign_omic_matrix`` behaviour tests live in ``test_omics.py`` since the
helper was split into ``molir/_omics.py`` so the registered predictors could
reach it without importing ``pytorch_lightning``; what remains of it here is the
guard on the compatibility re-export ``utils.py`` still exposes.

The loader construction that used to be tested here moved to the shared
``literature/_omics_loaders.py``, which MOLIR and SuperFELTR both call; its
behaviour is pinned through ``train_superfeltr_model`` in
``superfeltr/test_utils.py``, since that is the public entry point reaching it
without a Lightning fit per case.
"""

from __future__ import annotations

import numpy as np

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
