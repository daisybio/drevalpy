"""Each conformance check must accept a good component and reject its exact defect."""

from __future__ import annotations

import inspect
from typing import ClassVar

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.registry import predictor, sample_featurizer
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
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import DrugFeatureSource, SampleFeatureSource

BLOCK = "probe"


class GoodFeaturizer(SampleFeaturizer):
    """Conforming featurizer."""

    entity_id_only: ClassVar[bool] = True
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec(BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def __init__(self, *, n_features: int = 4) -> None:
        """Initialise the stub."""
        self._n_features = n_features
        self._offset = 0.0

    def _fit(self, source, **kwargs):
        entity_ids = kwargs.get("entity_ids")
        self._offset = float(len(entity_ids)) if entity_ids is not None else 0.0
        return self

    def _transform_features(self, source, entity_ids):
        values = np.arange(len(entity_ids) * self._n_features, dtype=np.float32)
        values = values.reshape(len(entity_ids), self._n_features) + self._offset
        return FeaturizerResult(entity_ids, {BLOCK: numeric_feature_block(values)}, (BLOCK,))

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
        """Initialise the stub."""
        super().__init__(n_features=n_features)


class MisreportsOutputDimFeaturizer(GoodFeaturizer):
    """Defect: ``output_dim`` disagrees with the produced width."""

    @property
    def output_dim(self) -> int:
        return self._n_features + 1


class MisalignedBlockFeaturizer(GoodFeaturizer):
    """Defect: emits fewer rows than there are entities."""

    def _transform_features(self, source, entity_ids):
        result = super()._transform_features(source, entity_ids)
        return FeaturizerResult(
            result.entity_ids[:-1], {BLOCK: numeric_feature_block(result.blocks[BLOCK].values[:-1])}, (BLOCK,)
        )


class EmptyBlocksFeaturizer(GoodFeaturizer):
    """Defect: produces no blocks at all."""

    def _transform_features(self, source, entity_ids):
        return FeaturizerResult(np.asarray(entity_ids), {}, ())


class ForgetfulStateFeaturizer(GoodFeaturizer):
    """Defect: ``get_state`` omits an attribute ``_transform_features`` reads."""

    def get_state(self) -> dict[str, object]:
        return {"n_features": self._n_features, "offset": 0.0}


class GoodPredictor(MatrixPredictor):
    """Conforming predictor: ordinary least squares with honest state."""

    def __init__(self, hyperparameters: dict[str, object] | None = None) -> None:
        """Initialise the stub."""
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
        """Initialise the stub."""
        super().__init__({"alpha": alpha})


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_dataset()


@pytest.fixture(scope="module")
def batch(dataset):
    return build_synthetic_batch(dataset)


def test_feature_source_for_matches_the_featurizer_side(dataset):
    assert isinstance(feature_source_for(GoodFeaturizer, dataset), SampleFeatureSource)
    source = feature_source_for(GoodDrugFeaturizer, dataset)
    assert isinstance(source, DrugFeatureSource)
    assert len(source.identifiers) == len(dataset.drug_ids)


@pytest.mark.parametrize("check", FEATURIZER_CHECKS)
@pytest.mark.parametrize("cls", [GoodFeaturizer, GoodDrugFeaturizer])
def test_a_conforming_featurizer_passes_every_check(check, cls, dataset):
    check(cls, dataset)


@pytest.mark.parametrize("check", PREDICTOR_CHECKS)
def test_a_conforming_predictor_passes_every_check(check, batch):
    check(GoodPredictor, batch)


def test_checks_default_to_synthetic_data_forward_kwargs_and_return_instances(dataset):
    check_featurizer_fit_transform(GoodFeaturizer)
    check_featurizer_fit_transform(GoodFeaturizer, dataset, n_features=7)
    check_predictor_fit_predict(GoodPredictor)
    assert isinstance(check_featurizer_instantiates(GoodFeaturizer), GoodFeaturizer)
    assert isinstance(check_predictor_instantiates(GoodPredictor), GoodPredictor)


@pytest.mark.parametrize(
    ("check", "cls", "error", "match"),
    [
        (check_featurizer_instantiates, RequiresAnArgumentFeaturizer, ConformanceError, "could not be constructed"),
        (check_featurizer_instantiates, dict, ConformanceError, "not a Featurizer"),
        (check_featurizer_fit_transform, MisreportsOutputDimFeaturizer, ConformanceError, "output_dim"),
        (check_featurizer_fit_transform, MisalignedBlockFeaturizer, ValueError, "must preserve requested entity order"),
        (check_featurizer_fit_transform, EmptyBlocksFeaturizer, ConformanceError, "returned no blocks"),
        (check_featurizer_state_round_trip, ForgetfulStateFeaturizer, ConformanceError, "different features"),
    ],
)
def test_featurizer_defects_are_rejected(check, cls, error, match, dataset):
    with pytest.raises(error, match=match):
        check(cls, dataset)


@pytest.mark.parametrize(
    ("check", "cls", "match"),
    [
        (check_predictor_instantiates, RequiresAnArgumentPredictor, "could not be constructed"),
        (check_predictor_instantiates, dict, "not a Predictor"),
        (check_predictor_fit_predict, NeverFitsPredictor, "non-finite"),
        (check_predictor_state_round_trip, ForgetfulStatePredictor, "predicted differently"),
        (check_predictor_state_round_trip, NeverFitsPredictor, "before the round trip"),
    ],
)
def test_predictor_defects_are_rejected(check, cls, match, batch):
    with pytest.raises(ConformanceError, match=match):
        check(cls, batch)


def test_check_tuples_name_every_check_with_two_positional_arguments():
    assert set(FEATURIZER_CHECKS) == {
        check_featurizer_instantiates,
        check_featurizer_fit_transform,
        check_featurizer_state_round_trip,
    }
    assert set(PREDICTOR_CHECKS) == {
        check_predictor_instantiates,
        check_predictor_fit_predict,
        check_predictor_state_round_trip,
    }
    for check in FEATURIZER_CHECKS + PREDICTOR_CHECKS:
        kinds = [p.kind for p in inspect.signature(check).parameters.values()]
        assert kinds.count(inspect.Parameter.POSITIONAL_OR_KEYWORD) == 2


@pytest.mark.parametrize("check", FEATURIZER_CHECKS)
@pytest.mark.parametrize("name", ["identity", "tissue"])
def test_builtin_sample_featurizers_conform(check, name, dataset):
    check(sample_featurizer.get(name), dataset)


@pytest.mark.parametrize("check", PREDICTOR_CHECKS)
@pytest.mark.parametrize("name", ["ridge", "elasticNet", "knn"])
def test_builtin_predictors_conform(check, name, batch):
    check(predictor.get(name), batch)
