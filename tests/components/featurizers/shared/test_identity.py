"""Tests for the shared one-hot identity featurizer.

Mirrors :mod:`drevalpy.components.featurizers.shared.identity`. One implementation
is bound to both entity sides by ``register_for_sides``, so the behavioural tests
are parameterized over the two generated classes and the registration assertions
check that each side got its own class with the right ``side`` stamped on.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.shared.identity import (
    CellLineIdentityFeaturizer,
    DrugIdentityFeaturizer,
    SharedIdentityFeaturizer,
)
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from tests.conftest import MockFeatureSource

_SIDE_CLASSES = [
    pytest.param(CellLineIdentityFeaturizer, id="cell-line"),
    pytest.param(DrugIdentityFeaturizer, id="drug"),
]


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES)
def test_identity_one_hot(featurizer_cls: type) -> None:
    featurizer = featurizer_cls()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e1", "e2", "e1"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)

    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 0] == 1.0


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES)
def test_identity_emits_category_metadata_block(featurizer_cls: type) -> None:
    featurizer = featurizer_cls()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e2", "e1"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert set(blocks) == {"identity", "identity_categories"}
    assert list(blocks["identity_categories"].values) == ["e1", "e2"]


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES)
def test_identity_round_trips_state(featurizer_cls: type) -> None:
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e1", "e2"], dtype=str)
    featurizer = featurizer_cls().fit(features, entity_ids=entity_ids)

    restored = featurizer_cls()
    restored.set_state(featurizer.get_state())

    assert restored.output_dim == 2
    np.testing.assert_allclose(
        restored.transform(features, entity_ids),
        featurizer.transform(features, entity_ids),
    )


def test_identity_registers_one_class_per_side() -> None:
    assert get_cell_line_featurizer("identity") is CellLineIdentityFeaturizer
    assert get_drug_featurizer("identity") is DrugIdentityFeaturizer
    assert CellLineIdentityFeaturizer is not DrugIdentityFeaturizer


def test_identity_side_is_stamped_per_binding() -> None:
    assert CellLineIdentityFeaturizer.side == "cell_line"
    assert DrugIdentityFeaturizer.side == "drug"


def test_identity_bindings_derive_from_the_shared_implementation() -> None:
    assert issubclass(CellLineIdentityFeaturizer, SharedIdentityFeaturizer)
    assert issubclass(DrugIdentityFeaturizer, SharedIdentityFeaturizer)
    assert SharedIdentityFeaturizer.side == ""
