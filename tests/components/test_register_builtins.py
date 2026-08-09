"""Tests for built-in component lazy registration maps."""

from __future__ import annotations

import drevalpy.components.core.plugins.register_builtins as register_builtins
from drevalpy.components.registry import get_predictor


def test_literature_predictors_register_from_split_modules() -> None:
    for name in ("precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego"):
        register_builtins.ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name


def test_naive_predictors_register_from_package() -> None:
    for name in ("naiveMean", "naiveDrugMean", "naiveMeanEffects"):
        register_builtins.ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name
