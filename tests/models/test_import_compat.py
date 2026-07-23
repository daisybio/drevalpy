"""Root export compatibility tests (deep model imports are unsupported)."""

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


@pytest.mark.parametrize("symbol", [name for name in models_all if name != "DRPModel"])
def test_root_symbol_exportable(symbol: str) -> None:
    module = importlib.import_module("drevalpy.models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        value = getattr(module, symbol)
    if symbol in {"construct_model", "MULTI_DRUG_MODEL_FACTORY", "SINGLE_DRUG_MODEL_FACTORY", "MODEL_FACTORY"}:
        assert value is not None
        return
    factory = _model_factory()
    assert value is factory.get(symbol) or value.get_model_name() in factory


@pytest.mark.parametrize("model_name", sorted(_model_factory()))
def test_model_factory_models_instantiate(model_name: str) -> None:
    model = _model_factory()[model_name]()
    assert model.get_model_name() == model_name
