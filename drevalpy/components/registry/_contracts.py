"""Attach feature contracts during component registration."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat, normalize_feature_contract


def _contract_defined_on_class(cls: type[Any], *attr_names: str) -> bool:
    return any(name in cls.__dict__ for name in attr_names)


def apply_registration_contracts(
    cls: type[Any],
    *,
    contract: FeatureContract | FeatureFormat | None = None,
    cell_line_contract: FeatureContract | FeatureFormat | None = None,
    drug_contract: FeatureContract | FeatureFormat | None = None,
) -> None:
    """Attach normalized contracts to *cls* from decorator arguments."""
    if contract is not None:
        if _contract_defined_on_class(cls, "contract"):
            msg = f"{cls.__name__!r} already defines a featurizer contract on the class body"
            raise ValueError(msg)
        cls.contract = normalize_feature_contract(contract)  # type: ignore[attr-defined]
    if cell_line_contract is not None:
        if _contract_defined_on_class(cls, "cell_line_contract"):
            msg = f"{cls.__name__!r} already defines a cell-line contract on the class body"
            raise ValueError(msg)
        cls.cell_line_contract = normalize_feature_contract(cell_line_contract)  # type: ignore[attr-defined]
    if drug_contract is not None:
        if _contract_defined_on_class(cls, "drug_contract"):
            msg = f"{cls.__name__!r} already defines a drug contract on the class body"
            raise ValueError(msg)
        cls.drug_contract = normalize_feature_contract(drug_contract)  # type: ignore[attr-defined]
