"""Tests for :mod:`drevalpy.testing.conformance`.

Each check is asserted twice: once against a conforming component, and once
against one with exactly the defect the check exists to find. A check that only
ever passes is worthless, so the negative cases carry the weight here.

The fixture components are declared locally rather than taken from the
registries. Registration is not needed to run a check, and a locally declared
class is what lets a single defect be introduced in isolation.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.testing.batch import build_synthetic_batch
from drevalpy.testing.conformance import (
    FEATURIZER_CHECKS,
    PREDICTOR_CHECKS,
    ConformanceError,
    check_featurizer_fit_transform,
    check_featurizer_instantiates,
    check_featurizer_state_round_trip,
    check_predictor_fit_predict,
    check_predictor_instantiates,
    check_predictor_state_round_trip,
    feature_source_for,
)
from drevalpy.testing.synthetic import build_synthetic_dataset
from drevalpy.types.data.batch.feature_block import BlockSpec, numeric_feature_block
from drevalpy.types.data.feature_source import CellLineFeatureSource, DrugFeatureSource

BLOCK = "probe"


class GoodFeaturizer(CellLineFeaturizer):
    """Conforming featurizer: hashed features, no view, honest state."""

    entity_id_only: ClassVar[bool] = True
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec(BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def __init__(self, *, n_features: int = 4) -> None:
        """Store the feature width and reset the fitted offset."""
        self._n_features = n_features
        self._offset = 0.0

    def _fit(self, source, **kwargs):
        entity_ids = kwargs.get("entity_ids")
        self._offset = float(len(entity_ids)) if entity_ids is not None else 0.0
        return self

    def _transform_blocks(self, source, entity_ids):
        values = np.arange(len(entity_ids) * self._n_features, dtype=np.float32)
        values = values.reshape(len(entity_ids), self._n_features) + self._offset
        return {BLOCK: numeric_feature_block(values)}

    @property
    def output_dim(self) -> int:
        return self._n_features

    def get_state(self) -> dict[str, object]:
        return {"n_features": self._n_features, "offset": self._offset}

    def set_state(self, state: dict[str, object]) -> None:
        self._n_features = int(state["n_features"])
        self._offset = float(state["offset"])


class GoodDrugFeaturizer(GoodFeaturizer, DrugFeaturizer):
    """Same logic on the drug side, so ``side`` dispatch can be exercised."""

    side: ClassVar[str] = "drug"


class RequiresAnArgumentFeaturizer(GoodFeaturizer):
    """Defect: not constructible with defaults."""

    def __init__(self, n_features: int) -> None:
        """Require *n_features* positionally, which the check must reject."""
        super().__init__(n_features=n_features)


class MisreportsOutputDimFeaturizer(GoodFeaturizer):
    """Defect: ``output_dim`` disagrees with the produced width."""

    @property
    def output_dim(self) -> int:
        return self._n_features + 1


class MisalignedBlockFeaturizer(GoodFeaturizer):
    """Defect: emits fewer rows than there are entities."""

    def _transform_blocks(self, source, entity_ids):
        blocks = super()._transform_blocks(source, entity_ids)
        return {BLOCK: numeric_feature_block(blocks[BLOCK].values[:-1])}


class EmptyBlocksFeaturizer(GoodFeaturizer):
    """Defect: produces no blocks at all."""

    def _transform_blocks(self, source, entity_ids):
        return {}


class ForgetfulStateFeaturizer(GoodFeaturizer):
    """Defect: ``get_state`` omits an attribute ``_transform_blocks`` reads."""

    def get_state(self) -> dict[str, object]:
        return {"n_features": self._n_features, "offset": 0.0}


class GoodPredictor(MatrixPredictor):
    """Conforming predictor: ordinary least squares with honest state."""

    def __init__(self, hyperparameters: dict[str, object] | None = None) -> None:
        """Reset the fitted coefficients."""
        super().__init__(hyperparameters)
        self._coefficients: np.ndarray | None = None

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        self._coefficients = np.linalg.lstsq(_with_intercept(x), y, rcond=None)[0]

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        if self._coefficients is None:
            return np.full(len(x), np.nan)
        return _with_intercept(x) @ self._coefficients

    def get_state(self) -> dict[str, object]:
        return {"coefficients": None if self._coefficients is None else self._coefficients.tolist()}

    def set_state(self, state: dict[str, object]) -> None:
        coefficients = state.get("coefficients")
        self._coefficients = None if coefficients is None else np.asarray(coefficients, dtype=np.float64)


def _with_intercept(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1), dtype=x.dtype)])


class NeverFitsPredictor(GoodPredictor):
    """Defect: ``_fit`` does nothing, so predictions stay NaN."""

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        return None


class ForgetfulStatePredictor(GoodPredictor):
    """Defect: ``get_state`` drops the fitted coefficients."""

    def get_state(self) -> dict[str, object]:
        return {"trained": True}

    def set_state(self, state: dict[str, object]) -> None:
        return None


class RequiresAnArgumentPredictor(GoodPredictor):
    """Defect: not constructible with defaults."""

    def __init__(self, alpha: float) -> None:
        """Require *alpha* positionally, which the check must reject."""
        super().__init__({"alpha": alpha})


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_dataset()


@pytest.fixture(scope="module")
def batch(dataset):
    return build_synthetic_batch(dataset)


class TestFeatureSourceFor:
    def test_a_cell_line_featurizer_gets_a_cell_line_source(self, dataset):
        assert isinstance(feature_source_for(GoodFeaturizer, dataset), CellLineFeatureSource)

    def test_a_drug_featurizer_gets_a_drug_source(self, dataset):
        assert isinstance(feature_source_for(GoodDrugFeaturizer, dataset), DrugFeatureSource)

    def test_the_source_exposes_the_matching_identifiers(self, dataset):
        source = feature_source_for(GoodDrugFeaturizer, dataset)

        assert len(source.identifiers) == len(dataset.drug_ids)


class TestFeaturizerChecksAcceptConformingComponents:
    @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
    @pytest.mark.parametrize("cls", [GoodFeaturizer, GoodDrugFeaturizer])
    def test_a_conforming_featurizer_passes_every_check(self, check, cls, dataset):
        check(cls, dataset)

    def test_the_checks_default_to_the_synthetic_dataset(self):
        """A plugin with ``entity_id_only`` featurizers needs no dataset of its own."""
        check_featurizer_fit_transform(GoodFeaturizer)

    def test_constructor_kwargs_are_forwarded(self, dataset):
        check_featurizer_fit_transform(GoodFeaturizer, dataset, n_features=7)

    def test_instantiation_returns_the_instance(self):
        assert isinstance(check_featurizer_instantiates(GoodFeaturizer), GoodFeaturizer)


class TestFeaturizerChecksRejectDefects:
    def test_a_non_default_constructible_featurizer_is_rejected(self):
        with pytest.raises(ConformanceError, match="could not be constructed"):
            check_featurizer_instantiates(RequiresAnArgumentFeaturizer)

    def test_a_non_featurizer_class_is_rejected(self):
        with pytest.raises(ConformanceError, match="not a Featurizer"):
            check_featurizer_instantiates(dict)  # type: ignore[arg-type]

    def test_a_wrong_output_dim_is_rejected(self, dataset):
        with pytest.raises(ConformanceError, match="output_dim"):
            check_featurizer_fit_transform(MisreportsOutputDimFeaturizer, dataset)

    def test_a_misaligned_block_is_rejected(self, dataset):
        with pytest.raises(ConformanceError, match="rows for"):
            check_featurizer_fit_transform(MisalignedBlockFeaturizer, dataset)

    def test_producing_no_blocks_is_rejected(self, dataset):
        with pytest.raises(ConformanceError, match="returned no blocks"):
            check_featurizer_fit_transform(EmptyBlocksFeaturizer, dataset)

    def test_an_incomplete_get_state_is_rejected(self, dataset):
        with pytest.raises(ConformanceError, match="different features"):
            check_featurizer_state_round_trip(ForgetfulStateFeaturizer, dataset)


class TestPredictorChecksAcceptConformingComponents:
    @pytest.mark.parametrize("check", PREDICTOR_CHECKS)
    def test_a_conforming_predictor_passes_every_check(self, check, batch):
        check(GoodPredictor, batch)

    def test_the_checks_default_to_the_synthetic_batch(self):
        check_predictor_fit_predict(GoodPredictor)

    def test_instantiation_returns_the_instance(self):
        assert isinstance(check_predictor_instantiates(GoodPredictor), GoodPredictor)


class TestPredictorChecksRejectDefects:
    def test_a_non_default_constructible_predictor_is_rejected(self):
        with pytest.raises(ConformanceError, match="could not be constructed"):
            check_predictor_instantiates(RequiresAnArgumentPredictor)

    def test_a_non_predictor_class_is_rejected(self):
        with pytest.raises(ConformanceError, match="not a Predictor"):
            check_predictor_instantiates(dict)  # type: ignore[arg-type]

    def test_a_predictor_that_never_trains_is_rejected(self, batch):
        with pytest.raises(ConformanceError, match="non-finite"):
            check_predictor_fit_predict(NeverFitsPredictor, batch)

    def test_an_incomplete_get_state_is_rejected(self, batch):
        with pytest.raises(ConformanceError, match="predicted differently"):
            check_predictor_state_round_trip(ForgetfulStatePredictor, batch)

    def test_a_predictor_that_never_trains_fails_the_round_trip_early(self, batch):
        """A predictor that cannot predict finitely cannot be compared at all."""
        with pytest.raises(ConformanceError, match="before the round trip"):
            check_predictor_state_round_trip(NeverFitsPredictor, batch)


class TestChecksCoverEveryPublicCheck:
    """The tuples exist so a plugin's suite cannot silently miss a new check."""

    def test_featurizer_checks_names_every_featurizer_check(self):
        assert set(FEATURIZER_CHECKS) == {
            check_featurizer_instantiates,
            check_featurizer_fit_transform,
            check_featurizer_state_round_trip,
        }

    def test_predictor_checks_names_every_predictor_check(self):
        assert set(PREDICTOR_CHECKS) == {
            check_predictor_instantiates,
            check_predictor_fit_predict,
            check_predictor_state_round_trip,
        }

    @pytest.mark.parametrize("check", FEATURIZER_CHECKS + PREDICTOR_CHECKS)
    def test_every_check_takes_the_same_two_positional_arguments(self, check):
        """What makes ``parametrize("check", FEATURIZER_CHECKS)`` work at all."""
        import inspect

        positional = [
            name
            for name, parameter in inspect.signature(check).parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]

        assert len(positional) == 2


class TestChecksAgainstShippedComponents:
    """The shipped components are the checks' own regression test.

    A check that rejects ``identity`` or ``ridge`` is wrong about the contract
    rather than right about the component.
    """

    @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
    @pytest.mark.parametrize("name", ["identity", "tissue", "constant"])
    def test_builtin_cell_line_featurizers_conform(self, check, name, dataset):
        from drevalpy.registry import cell_line_featurizer

        check(cell_line_featurizer.get(name), dataset)

    @pytest.mark.parametrize("check", PREDICTOR_CHECKS)
    @pytest.mark.parametrize("name", ["ridge", "lasso", "elasticNet", "knn"])
    def test_builtin_predictors_conform(self, check, name, batch):
        from drevalpy.registry import predictor

        check(predictor.get(name), batch)
