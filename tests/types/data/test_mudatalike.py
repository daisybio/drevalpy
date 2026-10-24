"""Tests for the ``MuDataLike`` protocol: ``isinstance`` enforces member presence, not signatures."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data import mudatalike
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.mudatalike import MuDataLike
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


class _MinimalDataset:
    @property
    def sample_ids(self) -> np.ndarray:
        return np.array(["cl1", "cl2"])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array(["d1"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.array([[1.0], [np.nan]])

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Lung"] * len(ids))

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        return ["relevance_score"]

    def get_response_layer(self, name: str) -> np.ndarray:
        return np.array([[9.0], [np.nan]])


class _MissingGetTissue(_MinimalDataset):
    get_tissue = None  # type: ignore[assignment]


class _MissingGetResponseLayer(_MinimalDataset):
    get_response_layer = None  # type: ignore[assignment]


class TestProtocolConformance:
    def test_dataset_satisfies_the_protocol(self):
        assert isinstance(synthetic_mudataset_gene_expression_fingerprints(), MuDataLike)
        assert MuDataLike in Dataset.__mro__

    def test_a_hand_built_stand_in_satisfies_the_protocol(self):
        dataset = _MinimalDataset()

        assert isinstance(dataset, MuDataLike)
        assert np.isnan(dataset.response_matrix).sum() == 1
        assert len(dataset.get_tissue(dataset.sample_ids)) == 2

    @pytest.mark.parametrize("candidate", [_MissingGetTissue(), _MissingGetResponseLayer(), object()])
    def test_an_object_missing_a_member_does_not_satisfy_the_protocol(self, candidate):
        assert not isinstance(candidate, MuDataLike)


class TestProtocolLimits:
    def test_the_protocol_cannot_be_instantiated(self):
        construct = getattr(mudatalike, "MuDataLike")  # noqa: B009 - defeats static resolution on purpose

        with pytest.raises(TypeError):
            construct()

    def test_isinstance_does_not_check_signatures(self):
        class WrongSignatures:
            sample_ids = "not an array"
            drug_ids = "not an array"
            response_matrix = "not an array"
            response_layer_names = "not a method"

            def get_tissue(self) -> None:
                return None

            def sample_column(self) -> None:
                return None

            def get_response_layer(self) -> None:
                return None

        assert isinstance(WrongSignatures(), MuDataLike)
