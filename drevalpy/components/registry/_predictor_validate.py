"""Class-level invariants for registered predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureFormat


def _leaf_interface_match(name: str, cls: type[Any], leaf_bases: tuple[type[Any], ...]) -> type[Any]:
    matches = [base for base in leaf_bases if issubclass(cls, base)]
    if len(matches) != 1:
        msg = (
            f"Predictor {name!r} must inherit exactly one of "
            f"FeatureFreePredictor, MatrixPredictor, BlockPredictor; "
            f"matched={[base.__name__ for base in matches]}"
        )
        raise ValueError(msg)
    return matches[0]


def _validate_matrix_contracts(name: str, cls: type[Any]) -> None:
    cell_line = getattr(cls, "cell_line_contract", None)
    drug = getattr(cls, "drug_contract", None)
    for side, contract in (("cell_line", cell_line), ("drug", drug)):
        if contract is None or contract.format != FeatureFormat.NUMERIC_MATRIX:
            actual = getattr(contract, "format", None)
            actual_value = actual.value if actual is not None else "<missing>"
            msg = f"Matrix predictor {name!r} requires numeric_matrix " f"{side} contract, got {actual_value!r}"
            raise ValueError(msg)


def validate_predictor_registration(name: str, cls: type[Any]) -> None:
    """Raise ``ValueError`` if a predictor class violates registration invariants.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Predictor class with contracts already attached by the decorator.
    """
    from drevalpy.components.hyperparameter_space import validate_component_hyperparameter_space
    from drevalpy.components.predictors.block import BlockPredictor
    from drevalpy.components.predictors.feature_free import FeatureFreePredictor
    from drevalpy.components.predictors.matrix import MatrixPredictor

    leaf_bases = (FeatureFreePredictor, MatrixPredictor, BlockPredictor)
    leaf_base = _leaf_interface_match(name, cls, leaf_bases)
    if leaf_base is MatrixPredictor:
        _validate_matrix_contracts(name, cls)
    validate_component_hyperparameter_space(name, cls)
