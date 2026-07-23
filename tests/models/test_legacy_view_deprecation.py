"""Contracts for deprecated flat view hyperparameter translation."""

from __future__ import annotations

import warnings

from drevalpy._deprecations import reset_deprecation_warnings
from drevalpy.components.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.models.config import ModelConfig
from drevalpy.models.flat_hyperparameters import apply_public_flat_hyperparameters
from drevalpy.models.zoo import get_zoo_config, zoo_model_config


def test_apply_public_flat_warns_on_view_keys() -> None:
    reset_deprecation_warnings()
    config = get_zoo_config("ElasticNet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updated = apply_public_flat_hyperparameters(
            config,
            {"cell_line_views": ["proteomics"], "drug_views": ["fingerprints"]},
        )
    assert updated.cell_line_featurizer is not None
    assert updated.cell_line_featurizer.name == "normalizedProteomics"
    assert any(
        issubclass(w.category, FutureWarning) and "Legacy cell_line_views/drug_views" in str(w.message) for w in caught
    )


def test_apply_public_flat_can_suppress_view_warning() -> None:
    reset_deprecation_warnings()
    config = get_zoo_config("ElasticNet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_public_flat_hyperparameters(
            config,
            {"cell_line_views": ["gene_expression"]},
            warn_legacy_view_keys=False,
        )
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


def test_public_hyperparameters_omit_view_keys_by_default() -> None:
    config = get_zoo_config("ElasticNet")
    public = public_hyperparameters_from_config(config)
    assert "cell_line_views" not in public
    assert "drug_views" not in public

    legacy = public_hyperparameters_from_config(config, include_view_keys=True)
    assert legacy["cell_line_views"] == ["gene_expression"]
    assert legacy["drug_views"] == ["fingerprints"]


def test_structured_model_config_path_does_not_warn() -> None:
    reset_deprecation_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composed = ModelConfig.from_spec("ElasticNet").create_model()
        assert composed is not None
    assert not any(
        issubclass(w.category, FutureWarning) and "Legacy cell_line_views/drug_views" in str(w.message) for w in caught
    )


def test_zoo_model_config_with_view_override_still_translates() -> None:
    reset_deprecation_warnings()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", FutureWarning)
        config = zoo_model_config(
            "ElasticNet",
            {"cell_line_views": ["gene_expression"], "alpha": 0.1},
        )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "scaledGeneExpression"
    assert "cell_line_views" not in config.predictor.hyperparameters
    assert config.predictor.hyperparameters["alpha"] == 0.1
