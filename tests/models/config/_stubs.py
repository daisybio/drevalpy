"""Stub component registrations shared by the ``models/config`` tests.

Config validation is mostly about what the registries say about a component, not
about what the component computes, so almost every test here first registers a
throwaway featurizer or predictor whose only real content is its contract. Those
registrations were written out per test, which is what made ``test_validation.py``
and ``test_block_specs.py`` read as near-clones of each other.

Plain ``_``-prefixed module, per the test-layout rules in ``AGENTS.md``: the
underscore keeps it out of collection, and the mirror policy walks ``drevalpy/``
only, so no mirrored test is demanded for it.

Every function here registers into the process-global component registries, so a
test using them must be under ``isolated_component_registries`` from
``tests/registry/_helpers.py``.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import PredictionMode
from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer
from drevalpy.registry.drug_featurizer import register as register_drug_featurizer
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.feature_block import BlockSpec

REGRESSION_ONLY = frozenset({PredictionMode.REGRESSION})


def register_featurizer_stub(
    name: str,
    *,
    side: str,
    contract: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
    output_block_specs: tuple[BlockSpec, ...] | None = None,
) -> type:
    """Register a featurizer that declares a contract and computes nothing.

    :param name: Registry name.
    :param side: ``"cell_line"`` or ``"drug"``.
    :param contract: Output feature format the registration declares.
    :param output_block_specs: Declared block specs; omitted leaves the
        view-fallback path in play.
    :returns: The registered stub class.
    """
    register = register_cell_line_featurizer if side == "cell_line" else register_drug_featurizer

    @register(name, description=f"{side} stub", contract=contract)
    class Stub:
        pass

    if output_block_specs is not None:
        Stub.output_block_specs = output_block_specs
    return Stub


def register_matrix_predictor_stub(
    name: str = "densePred",
    *,
    cell_line_contract: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
    drug_contract: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
) -> type:
    """Register a regression-only ``MatrixPredictor`` that predicts zeros.

    :param name: Registry name.
    :param cell_line_contract: Cell-line contract the registration declares.
    :param drug_contract: Drug contract the registration declares.
    :returns: The registered stub class.
    """

    @register_predictor(
        name,
        description="matrix stub",
        cell_line_contract=cell_line_contract,
        drug_contract=drug_contract,
    )
    class Stub(MatrixPredictor):
        supported_modes = REGRESSION_ONLY

        def _fit_matrix(self, x, y) -> None:
            return None

        def _predict_matrix(self, x):
            return np.zeros(len(x), dtype=np.float64)

    return Stub


def register_block_predictor_stub(
    name: str = "blockPred",
    *,
    cell_line_contract: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
    drug_contract: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
    required_cell_line_block_specs: tuple[BlockSpec, ...] | None = None,
    required_drug_block_specs: tuple[BlockSpec, ...] | None = None,
) -> type:
    """Register a regression-only ``BlockPredictor`` that predicts zeros.

    :param name: Registry name.
    :param cell_line_contract: Cell-line contract the registration declares.
    :param drug_contract: Drug contract the registration declares.
    :param required_cell_line_block_specs: Block specs the predictor demands.
    :param required_drug_block_specs: Block specs the predictor demands.
    :returns: The registered stub class.
    """

    @register_predictor(
        name,
        description="block stub",
        cell_line_contract=cell_line_contract,
        drug_contract=drug_contract,
    )
    class Stub(BlockPredictor):
        supported_modes = REGRESSION_ONLY

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    if required_cell_line_block_specs is not None:
        Stub.required_cell_line_block_specs = required_cell_line_block_specs
    if required_drug_block_specs is not None:
        Stub.required_drug_block_specs = required_drug_block_specs
    return Stub


def register_feature_free_predictor_stub(name: str = "naiveMean") -> type:
    """Register a ``FeatureFreePredictor`` that predicts zeros.

    :param name: Registry name.
    :returns: The registered stub class.
    """

    @register_predictor(
        name,
        description="feature-free stub",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class Stub(FeatureFreePredictor):
        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    return Stub


def register_dense_trio() -> None:
    """Register the dense cell-line / dense drug / matrix-predictor triple.

    The baseline every contract-mismatch test varies one member of.
    """
    register_featurizer_stub("denseCellLine", side="cell_line")
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")
