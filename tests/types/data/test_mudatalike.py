"""Tests for the ``MuDataLike`` protocol.

The protocol exists so splitters can be exercised against a hand-built stand-in
instead of a full ``Dataset``. It is ``runtime_checkable``, so these tests pin
what ``isinstance`` actually enforces -- member presence, not signatures -- to
keep callers from relying on a guarantee it does not give.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data import mudatalike
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.mudatalike import MuDataLike
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


class _MinimalDataset:
    """Smallest object satisfying the protocol: two entities, one measured pair."""

    @property
    def cell_line_ids(self) -> np.ndarray:
        return np.array(["cl1", "cl2"])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array(["d1"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.array([[1.0], [np.nan]])

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Lung"] * len(ids))

    def response_layer_names(self) -> list[str]:
        return ["relevance_score"]

    def get_response_layer(self, name: str) -> np.ndarray:
        if name != "relevance_score":
            raise KeyError(name)
        return np.array([[9.0], [np.nan]])


class _MissingGetTissue:
    """Satisfies every member but the tissue lookup."""

    @property
    def cell_line_ids(self) -> np.ndarray:
        return np.array(["cl1"])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array(["d1"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.array([[1.0]])

    def response_layer_names(self) -> list[str]:
        return []

    def get_response_layer(self, name: str) -> np.ndarray:
        raise KeyError(name)


class _MissingGetResponseLayer:
    """Satisfies every member but the layer accessor the quality filter needs."""

    @property
    def cell_line_ids(self) -> np.ndarray:
        return np.array(["cl1"])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array(["d1"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.array([[1.0]])

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Lung"] * len(ids))

    def response_layer_names(self) -> list[str]:
        return []


class TestProtocolConformance:
    def test_dataset_satisfies_the_protocol(self):
        dataset = synthetic_mudataset_gene_expression_fingerprints()

        assert isinstance(dataset, MuDataLike)

    def test_dataset_declares_the_protocol_as_a_base(self):
        assert MuDataLike in Dataset.__mro__

    def test_a_hand_built_stand_in_satisfies_the_protocol(self):
        assert isinstance(_MinimalDataset(), MuDataLike)

    def test_an_object_missing_a_member_does_not_satisfy_the_protocol(self):
        assert not isinstance(_MissingGetTissue(), MuDataLike)

    def test_an_object_missing_the_layer_accessor_does_not_satisfy_the_protocol(self):
        assert not isinstance(_MissingGetResponseLayer(), MuDataLike)

    def test_an_unrelated_object_does_not_satisfy_the_protocol(self):
        assert not isinstance(object(), MuDataLike)


class TestProtocolLimits:
    def test_the_protocol_cannot_be_instantiated(self):
        # Resolved by name at runtime so the static checker does not (correctly)
        # reject the deliberate protocol instantiation outright.
        construct = getattr(mudatalike, "MuDataLike")  # noqa: B009 - defeats static resolution on purpose

        with pytest.raises(TypeError):
            construct()

    def test_isinstance_does_not_check_signatures(self):
        """Only member presence is tested, never a signature.

        Callers must not read a successful ``isinstance`` as a signature guarantee.
        """

        class WrongSignatures:
            cell_line_ids = "not an array"
            drug_ids = "not an array"
            response_matrix = "not an array"
            response_layer_names = "not a method"

            def get_tissue(self) -> None:
                return None

            def get_response_layer(self) -> None:
                return None

        assert isinstance(WrongSignatures(), MuDataLike)


class TestStandInBehaviour:
    def test_response_matrix_carries_nan_for_unmeasured_pairs(self):
        dataset = _MinimalDataset()

        assert np.isnan(dataset.response_matrix).sum() == 1

    def test_get_tissue_returns_one_label_per_requested_id(self):
        dataset = _MinimalDataset()

        assert len(dataset.get_tissue(dataset.cell_line_ids)) == 2
