"""Tests for predictor class invariants enforced at registration."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config._predictor_traits import needs_identity_drug_routing
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcreteMatrix(MatrixPredictor):
    """Minimal concrete matrix predictor."""

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.float64)


class _ConcreteBlock(BlockPredictor):
    """Minimal concrete block predictor."""

    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


class _ConcreteFeatureFree(FeatureFreePredictor):
    """Minimal concrete feature-free predictor."""

    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def test_predictor_must_inherit_exactly_one_leaf_interface() -> None:
    with pytest.raises(ValueError, match="exactly one of"):

        @register_predictor(
            "noLeafPred",
            description="missing leaf",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class NoLeafPred:
            pass


def test_matrix_predictor_requires_numeric_contracts() -> None:
    with pytest.raises(ValueError, match="requires numeric_matrix cell_line contract"):

        @register_predictor(
            "graphMatrixPred",
            description="bad matrix",
            cell_line_contract=FeatureFormat.GRAPH,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class GraphMatrixPred(_ConcreteMatrix):
            pass


def test_single_drug_feature_predictor_needs_no_routing_declaration() -> None:
    @register_predictor(
        "plainSingleDrug",
        description="declares only its scope",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class PlainSingleDrug(_ConcreteMatrix):
        scope = ModelScope.SINGLE_DRUG

    assert predictor_registry.get("plainSingleDrug") is PlainSingleDrug
    assert needs_identity_drug_routing("plainSingleDrug") is True


def test_feature_free_single_drug_skips_identity_routing() -> None:
    @register_predictor(
        "freeSingleDrug",
        description="feature free single drug",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class FreeSingleDrug(_ConcreteFeatureFree):
        scope = ModelScope.SINGLE_DRUG

    assert predictor_registry.get("freeSingleDrug") is FreeSingleDrug
    assert needs_identity_drug_routing("freeSingleDrug") is False


def test_registered_predictor_scope_defaults_to_multi_drug() -> None:
    @register_predictor(
        "scopedPred",
        description="scope default",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class ScopedPred(_ConcreteBlock):
        pass

    assert ScopedPred.scope is ModelScope.MULTI_DRUG


def test_register_existing_rejects_invalid_predictor_class() -> None:
    class InvalidRestored:
        description = "invalid"
        tags: frozenset[str] = frozenset()
        reference = None
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    with pytest.raises(ValueError, match="exactly one of"):
        predictor_registry.register_existing("invalidRestored", InvalidRestored)


def test_register_existing_restores_valid_predictor() -> None:
    @register_predictor(
        "restorablePred",
        description="restorable",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class RestorablePred(_ConcreteBlock):
        pass

    predictor_registry.clear()
    assert predictor_registry.list_names() == []
    predictor_registry.register_existing("restorablePred", RestorablePred)
    assert predictor_registry.get("restorablePred") is RestorablePred


# ---------------------------------------------------------------------------
# Abstract-member rejection
# ---------------------------------------------------------------------------


def test_a_predictor_missing_its_subclass_hooks_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"does not implement _fit, _predict"):

        @register_predictor(
            "abstractPred",
            description="forgot _fit and _predict",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class AbstractPred(BlockPredictor):
            pass


def test_a_matrix_predictor_missing_its_matrix_hooks_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"does not implement _fit_matrix, _predict_matrix"):

        @register_predictor(
            "abstractMatrixPred",
            description="forgot the matrix hooks",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class AbstractMatrixPred(MatrixPredictor):
            pass


def test_the_rejection_names_the_registry_and_the_name() -> None:
    with pytest.raises(ValueError, match=r"predictor 'namedPred' \(NamedPred\)"):

        @register_predictor(
            "namedPred",
            description="names itself in the error",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class NamedPred(BlockPredictor):
            pass


def test_register_existing_also_rejects_an_abstract_class() -> None:
    class AbstractRestored(BlockPredictor):
        description = "abstract"
        tags: frozenset[str] = frozenset()
        reference = None
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    with pytest.raises(ValueError, match="does not implement"):
        predictor_registry.register_existing("abstractRestored", AbstractRestored)
