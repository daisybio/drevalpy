"""Class-level invariants for registered predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.components.predictors.inputs import predictor_inputs


def validate_predictor_registration(name: str, cls: type[Any]) -> None:
    """Raise if a predictor class violates registration invariants.

    Both sides must declare their inputs (``predictor_inputs`` raises ``TypeError``
    otherwise), and the hyperparameter space must be valid.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Predictor class.
    """
    for side in ("sample", "drug"):
        predictor_inputs(cls, side)
    validate_component_hyperparameter_space(name, cls)
