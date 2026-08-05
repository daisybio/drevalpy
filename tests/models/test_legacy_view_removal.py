"""Contracts for hard removal of flat view hyperparameter keys."""

from __future__ import annotations

import warnings

import pytest

from drevalpy._deprecations import reset_deprecation_warnings
from drevalpy.components.tuning.public_flat import (
    apply_public_hyperparameters_to_config,
    public_hyperparameters_from_config,
)
from drevalpy.models import construct_model
from drevalpy.models.zoo import get_zoo_config, zoo_model_config


def test_apply_public_flat_rejects_view_keys() -> None:
    config = get_zoo_config("ElasticNet")
    with pytest.raises(ValueError, match=r"Legacy view keys|no longer supported"):
        apply_public_hyperparameters_to_config(
            config,
            {"cell_line_views": ["proteomics"], "drug_views": ["fingerprints"]},
        )


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
        model = construct_model("ElasticNet")()
        assert model._stack is not None
    assert not any(
        issubclass(w.category, FutureWarning) and "Legacy cell_line_views/drug_views" in str(w.message) for w in caught
    )


def test_zoo_model_config_raises_on_view_keys() -> None:
    with pytest.raises(ValueError, match=r"Legacy view keys|no longer supported"):
        zoo_model_config(
            "ElasticNet",
            {"cell_line_views": ["gene_expression"], "alpha": 0.1},
        )
