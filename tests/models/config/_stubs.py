"""Stub component registrations shared by the ``models/config`` tests.

Config validation is mostly about what the registries say about a component, not
about what the component computes, so almost every test here first registers a
throwaway featurizer or predictor whose only real content is its feature format. Those
registrations were written out per test, which made the validation tests read as
near-clones of each other.

Plain ``_``-prefixed module: the underscore keeps it out of collection.

Every function here registers into the process-global component registries; the
autouse ``registry_snapshot`` fixture in ``tests/conftest.py`` restores them after each test.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry.featurizer import register as register_featurizer
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.enums.prediction_mode import PredictionMode

REGRESSION_ONLY = frozenset({PredictionMode.REGRESSION})


def register_featurizer_stub(
    name: str,
    *,
    side: str,
    output_format: FeatureFormat = FeatureFormat.NUMERIC_MATRIX,
    output_block_specs: tuple[BlockSpec, ...] | None = None,
) -> type:
    """Register a featurizer for one side that declares its output blocks and computes nothing.

    :param name: Registry name.
    :param side: ``"sample"`` or ``"drug"``.
    :param output_format: Format of the one block emitted when *output_block_specs* is omitted;
        the block is named after the configured view, else after *name*.
    :param output_block_specs: Declared block specs, emitted whatever the config.
    :returns: The registered stub class.
    """

    class Stub(Featurizer):
        def __init__(self, view: str | None = None, **kwargs) -> None:
            self._view = view

        @classmethod
        def output_blocks(cls, config) -> tuple[BlockSpec, ...]:
            if output_block_specs is not None:
                return output_block_specs
            view = config.get("view")
            return (BlockSpec(view if isinstance(view, str) else name, output_format),)

        def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
            return self

        def transform(self, source, ids) -> FeaturizerResult:
            return FeaturizerResult(ids, {}, ())

    return register_featurizer(name, side=side, description=f"{side} stub")(Stub)


def register_matrix_predictor_stub(name: str = "densePred") -> type:
    """Register a regression-only ``MatrixPredictor`` that predicts zeros.

    :param name: Registry name.
    :returns: The registered stub class.
    """

    @register_predictor(name, description="matrix stub")
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
    sample_blocks: tuple[BlockSpec, ...] = (),
    drug_blocks: tuple[BlockSpec, ...] = (),
) -> type:
    """Register a regression-only block predictor that predicts zeros.

    :param name: Registry name.
    :param sample_blocks: Sample blocks the predictor requires; none takes any blocks.
    :param drug_blocks: Drug blocks the predictor requires; none takes any blocks.
    :returns: The registered stub class.
    """

    @register_predictor(name, description="block stub")
    class Stub(Predictor):
        sample_inputs = Inputs.blocks(*sample_blocks)
        drug_inputs = Inputs.blocks(*drug_blocks)
        supported_modes = REGRESSION_ONLY

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    return Stub


def register_feature_free_predictor_stub(name: str = "naiveMean") -> type:
    """Register a ``FeatureFreePredictor`` that predicts zeros.

    :param name: Registry name.
    :returns: The registered stub class.
    """

    @register_predictor(name, description="feature-free stub")
    class Stub(FeatureFreePredictor):
        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return np.zeros(batch.n_pairs, dtype=np.float64)

    return Stub


def register_dense_trio() -> None:
    """Register the dense sample / dense drug / matrix-predictor triple.

    The baseline every format-mismatch test varies one member of.
    """
    register_featurizer_stub("denseSample", side="sample")
    register_featurizer_stub("denseDrug", side="drug")
    register_matrix_predictor_stub("densePred")
