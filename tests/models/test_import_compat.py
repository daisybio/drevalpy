"""Root export compatibility tests (named model symbols are unsupported)."""

from __future__ import annotations

import importlib
import warnings

import pytest

from drevalpy.models import __all__ as models_all


def _model_factory():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from drevalpy.models import MODEL_FACTORY

        return MODEL_FACTORY


def test_models_all_exports_expected_symbols() -> None:
    assert set(models_all) == {
        "DRPModel",
        "construct_model",
        "MULTI_DRUG_MODEL_FACTORY",
        "SINGLE_DRUG_MODEL_FACTORY",
        "MODEL_FACTORY",
    }


@pytest.mark.parametrize("symbol", list(models_all))
def test_root_symbol_exportable(symbol: str) -> None:
    module = importlib.import_module("drevalpy.models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        value = getattr(module, symbol)
    assert value is not None


@pytest.mark.parametrize("model_name", sorted(_model_factory()))
def test_model_factory_models_instantiate(model_name: str) -> None:
    model = _model_factory()[model_name]()
    assert model.get_model_name() == model_name


def test_named_model_exports_removed() -> None:
    module = importlib.import_module("drevalpy.models")
    with pytest.raises(AttributeError):
        _ = module.ElasticNetModel
