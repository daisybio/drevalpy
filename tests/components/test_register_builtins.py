"""Tests for built-in component lazy registration maps."""

from __future__ import annotations

from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor


def test_literature_predictors_register_from_split_modules() -> None:
    for name in ("precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego"):
        ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name


def test_naive_predictors_register_from_package() -> None:
    for name in ("naiveMean", "naiveDrugMean", "naiveMeanEffects"):
        ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name
