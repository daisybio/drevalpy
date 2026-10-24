"""Tests for feature contracts, hyperparameter-space defaults and the training context."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.contracts import (
    FeatureContract,
    FeatureFormat,
    contracts_compatible,
    featurizer_contract,
    normalize_feature_contract,
    predictor_contracts,
)
from drevalpy.components.contracts.hyperparameter_space import (
    TunableComponentMixin,
    validate_component_hyperparameter_space,
    validate_hyperparameter_space,
)
from drevalpy.components.contracts.training_context import TrainingContext, check_n_jobs
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.errors import InvalidInputError
from drevalpy.models.config import FeaturizerConfig, PredictorConfig
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer
from tests.registry._helpers import isolated_component_registries


@pytest.mark.parametrize(
    ("produced", "required", "compatible"),
    [
        (FeatureFormat.NUMERIC_MATRIX, FeatureFormat.NUMERIC_MATRIX, True),
        (FeatureFormat.GRAPH, FeatureFormat.GRAPH, True),
        (FeatureFormat.RAGGED_SEQUENCE, FeatureFormat.RAGGED_SEQUENCE, True),
        (FeatureFormat.GRAPH, FeatureFormat.NUMERIC_MATRIX, False),
    ],
)
def test_contracts_are_compatible_by_format_only(produced, required, compatible: bool) -> None:
    assert contracts_compatible(FeatureContract(format=produced), FeatureContract(format=required)) is compatible


def test_feature_contract_is_frozen_and_normalizes_the_format_shorthand() -> None:
    contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    with pytest.raises(AttributeError):
        cast(Any, contract).format = FeatureFormat.GRAPH
    assert normalize_feature_contract(FeatureFormat.GRAPH) == FeatureContract(format=FeatureFormat.GRAPH)


def test_featurizer_contract_reads_the_canonical_attribute_and_requires_it() -> None:
    class WithContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    class WithoutContract:
        pass

    assert featurizer_contract(WithContract).format == FeatureFormat.GRAPH
    with pytest.raises(TypeError, match="must define a contract"):
        featurizer_contract(WithoutContract)


def test_predictor_contracts_read_the_canonical_attributes_and_require_both() -> None:
    class WithContracts:
        sample_contract = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
        drug_contract = FeatureContract(format=FeatureFormat.GRAPH)

    class MissingDrugContract:
        sample_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    sample, drug = predictor_contracts(WithContracts)
    assert sample.format == FeatureFormat.RAGGED_SEQUENCE
    assert drug.format == FeatureFormat.GRAPH
    with pytest.raises(TypeError, match="must define both"):
        predictor_contracts(MissingDrugContract)


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_validate_hyperparameter_space_rejects_incomplete_specs_and_accepts_complete_ones() -> None:
    with pytest.raises(ValueError, match="missing 'default' for 'alpha'"):
        validate_hyperparameter_space({"alpha": {"type": "float", "low": 0.1, "high": 1.0}}, context="unit-test")
    with pytest.raises(ValueError, match="non-mapping specs for 'alpha'"):
        validate_hyperparameter_space({"alpha": 1.0}, context="unit-test")
    validate_hyperparameter_space({"alpha": {"type": "float", "default": 0.5}}, context="unit-test")

    class Ok:
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "default": 1.0}}

    validate_component_hyperparameter_space("ok", Ok)


def test_predictor_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_predictor(
            "badSpacePred",
            description="missing default",
            sample_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpacePred(MatrixPredictor):
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

            def _fit_matrix(self, x, y) -> None:
                return None

            def _predict_matrix(self, x):
                return x[:, 0]


def test_featurizer_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_sample_featurizer(
            "badSpaceFeat",
            description="missing default",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpaceFeat:
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"n_components": {"type": "int", "low": 8, "high": 64}}


def test_configs_reject_a_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        PredictorConfig(name="elasticNet", hyperparameter_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
    with pytest.raises(ValidationError, match="missing 'default'"):
        FeaturizerConfig(
            name="pca",
            view="expression",
            hyperparameter_space={"n_components": {"type": "int", "low": 8, "high": 64}},
        )


def test_both_component_bases_inherit_the_mixin_without_re_declaring_its_hooks() -> None:
    for component in (Featurizer, Predictor):
        assert issubclass(component, TunableComponentMixin)
        for hook in ("get_hyperparameter_space", "get_default_hyperparameters", "get_state", "set_state"):
            assert hook not in component.__dict__


def test_mixin_defaults_are_read_off_the_declared_space_and_validated() -> None:
    class Untunable(TunableComponentMixin):
        pass

    class Tunable(TunableComponentMixin):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.25}}

    class Incomplete(TunableComponentMixin):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

    assert Untunable.get_hyperparameter_space() == {}
    assert Untunable.get_default_hyperparameters() == {}
    assert Tunable.get_default_hyperparameters() == {"alpha": 0.25}
    with pytest.raises(ValueError, match=r"Incomplete.get_hyperparameter_space\(\)"):
        Incomplete.get_default_hyperparameters()


def test_the_default_state_is_empty_and_is_fitted_reads_the_state_hook() -> None:
    class Stateless(TunableComponentMixin):
        pass

    class Fitted(Predictor):
        def get_state(self) -> dict[str, object]:
            return {"weights": [1.0]}

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return None

    component = Stateless()
    assert component.get_state() == {}
    assert component.set_state({"ignored": 1}) is None
    assert component.get_state() == {}
    assert Fitted().is_fitted() is True


def test_default_checkpoint_dir_is_a_unique_temp_path_not_the_cwd() -> None:
    first, second = TrainingContext(), TrainingContext()

    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_dir.is_absolute()
    assert Path(tempfile.gettempdir()) in Path(str(first.checkpoint_dir)).parents
    assert not first.checkpoint_dir.exists()


def test_defaults_are_unshared_and_an_explicit_checkpoint_dir_is_kept(tmp_path) -> None:
    first, second = TrainingContext(), TrainingContext()
    first.logging_metadata["run"] = "1"

    assert second.logging_metadata == {}
    assert first.n_jobs == 1
    context = TrainingContext(checkpoint_dir=tmp_path, logging_metadata={"fold": "0"})
    assert context.checkpoint_dir == tmp_path
    assert context.logging_metadata == {"fold": "0"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.checkpoint_dir = tmp_path / "other"


@pytest.mark.parametrize("n_jobs", [1, 2, 16, -1])
def test_check_n_jobs_accepts_a_worker_count_or_all_cpus(n_jobs: int) -> None:
    check_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [0, -2, -8])
def test_check_n_jobs_rejects_everything_else(n_jobs: int) -> None:
    with pytest.raises(InvalidInputError, match="n_jobs must be a positive integer or -1"):
        check_n_jobs(n_jobs)
