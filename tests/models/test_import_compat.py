"""Root export compatibility tests (deep model imports are unsupported)."""

from __future__ import annotations

import importlib

import pytest

from drevalpy.models import MODEL_FACTORY, __all__ as models_all


@pytest.mark.parametrize("symbol", [name for name in models_all if name not in {"DRPModel", "SingleDrugModelMixin"}])
def test_root_symbol_exportable(symbol: str) -> None:
    module = importlib.import_module("drevalpy.models")
    value = getattr(module, symbol)
    if symbol in {"construct_model", "MULTI_DRUG_MODEL_FACTORY", "SINGLE_DRUG_MODEL_FACTORY", "MODEL_FACTORY"}:
        assert value is not None
        return
    assert value is MODEL_FACTORY.get(symbol) or value.get_model_name() in MODEL_FACTORY


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_model_factory_models_instantiate(model_name: str) -> None:
    model = MODEL_FACTORY[model_name]()
    assert model.get_model_name() == model_name
