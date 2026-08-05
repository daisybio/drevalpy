"""Attach feature contracts during component registration."""

from __future__ import annotations

from typing import Any, Protocol

from drevalpy.components.contracts import FeatureContract, FeatureFormat, normalize_feature_contract


class FeaturizerContractAttributes(Protocol):
    """Class variables attached to featurizers during registration."""

    contract: FeatureContract


class PredictorContractAttributes(Protocol):
    """Class variables attached to predictors during registration."""

    cell_line_contract: FeatureContract
    drug_contract: FeatureContract


def _set_class_attribute(cls: type[Any], name: str, value: object) -> None:
    """Assign a registration attribute on *cls* via ``setattr``.

    :param cls: Component class receiving the attribute.
    :param name: Attribute name to set.
    :param value: Attribute value to assign.
    """
    setattr(cls, name, value)


def _contract_defined_on_class(cls: type[Any], *attr_names: str) -> bool:
    return any(name in cls.__dict__ for name in attr_names)


def apply_registration_contracts(
    cls: type[Any],
    *,
    contract: FeatureContract | FeatureFormat | None = None,
    cell_line_contract: FeatureContract | FeatureFormat | None = None,
    drug_contract: FeatureContract | FeatureFormat | None = None,
) -> None:
    """Attach normalized contracts to *cls* from decorator arguments.

    :param cls: Class receiving registration contracts.
    :param contract: contract.
    :param cell_line_contract: cell line contract.
    :param drug_contract: drug contract.
    :raises ValueError: Raised on invalid input.
    """
    if contract is not None:
        if _contract_defined_on_class(cls, "contract"):
            msg = f"{cls.__name__!r} already defines a featurizer contract on the class body"
            raise ValueError(msg)
        _set_class_attribute(cls, "contract", normalize_feature_contract(contract))
    if cell_line_contract is not None:
        if _contract_defined_on_class(cls, "cell_line_contract"):
            msg = f"{cls.__name__!r} already defines a cell-line contract on the class body"
            raise ValueError(msg)
        _set_class_attribute(cls, "cell_line_contract", normalize_feature_contract(cell_line_contract))
    if drug_contract is not None:
        if _contract_defined_on_class(cls, "drug_contract"):
            msg = f"{cls.__name__!r} already defines a drug contract on the class body"
            raise ValueError(msg)
        _set_class_attribute(cls, "drug_contract", normalize_feature_contract(drug_contract))
