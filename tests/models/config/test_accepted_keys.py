"""Tests for the accepted-key helpers shared by key validation and the tuning modules."""

from __future__ import annotations

import pytest

from drevalpy.errors import UnknownNameError
from drevalpy.models.config import SampleFeaturizerConfig
from drevalpy.models.config._accepted_keys import featurizer_accepted_keys, leaf_selector, predictor_accepted_keys
from drevalpy.registry._builtins import register_builtin_components


@pytest.fixture(autouse=True)
def _registry() -> None:
    """Register the built-ins the helpers look classes up in."""
    register_builtin_components()


class TestPredictorAcceptedKeys:
    """Defaults, tunable space and ``non_tunable_hyperparameters`` all count as accepted."""

    def test_merges_defaults_and_space(self) -> None:
        class Predictor:
            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {"alpha": 1.0}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {"l1_ratio": {"default": 0.5}}

        assert predictor_accepted_keys(Predictor) == {"alpha", "l1_ratio"}  # type: ignore[arg-type]

    def test_includes_non_tunable_hyperparameters(self) -> None:
        class Predictor:
            non_tunable_hyperparameters = {"max_iter": 1000}

            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {"alpha": 1.0}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {}

        assert predictor_accepted_keys(Predictor) == {"alpha", "max_iter"}  # type: ignore[arg-type]

    def test_tolerates_an_absent_declaration(self) -> None:
        class Predictor:
            @staticmethod
            def get_default_hyperparameters() -> dict[str, object]:
                return {}

            @staticmethod
            def get_hyperparameter_space() -> dict[str, object]:
                return {}

        assert predictor_accepted_keys(Predictor) == set()  # type: ignore[arg-type]


class TestFeaturizerAcceptedKeys:
    """The accepted keys of a featurizer are its registered class's hyperparameter space."""

    def test_reads_the_space_of_the_registered_class(self) -> None:
        config = SampleFeaturizerConfig(name="pca", view="gene_expression")

        assert "n_components" in featurizer_accepted_keys(config, "sample")

    def test_rejects_an_unknown_side(self) -> None:
        config = SampleFeaturizerConfig(name="pca", view="gene_expression")

        with pytest.raises(UnknownNameError):
            featurizer_accepted_keys(config, "tissue")  # type: ignore[arg-type]


class TestLeafSelector:
    """The selector is the name, qualified by the view for view-parametric featurizers."""

    def test_qualifies_a_view_parametric_featurizer(self) -> None:
        config = SampleFeaturizerConfig(name="pca", view="methylation")

        assert leaf_selector(config) == "pca[methylation]"

    def test_leaves_a_plain_featurizer_unqualified(self) -> None:
        config = SampleFeaturizerConfig(name="identity")

        assert leaf_selector(config) == "identity"
