"""Feature kind and contract objects for component compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FeatureKind(StrEnum):
    """Format of featurizer outputs / predictor inputs."""

    DENSE = "dense"
    GRAPH = "graph"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class FeatureContract:
    """Structured description of a feature representation.

    Compatibility intentionally checks only the broad feature kind for now.
    Additional fields may be added later when real compatibility requirements
    appear; graph compatibility is therefore currently just ``graph`` expected
    and ``graph`` provided.
    """

    kind: FeatureKind


def normalize_feature_contract(contract: FeatureContract | FeatureKind) -> FeatureContract:
    """Return a ``FeatureContract`` from a contract object or kind shorthand.

    Args:
        contract: A ``FeatureContract`` instance or ``FeatureKind`` enum member.

    Returns:
        Normalized ``FeatureContract`` instance.

    Raises:
        TypeError: If *contract* is neither ``FeatureContract`` nor ``FeatureKind``.
    """
    if isinstance(contract, FeatureContract):
        return contract
    if isinstance(contract, FeatureKind):
        return FeatureContract(kind=contract)
    msg = f"Expected FeatureContract or FeatureKind, got {type(contract).__name__}"
    raise TypeError(msg)


def featurizer_contract(cls: type[Any]) -> FeatureContract:
    """Return the featurizer contract, preferring ``contract`` over legacy ``output_contract``.

    Args:
        cls: Featurizer class registered in the component registry.

    Returns:
        Resolved ``FeatureContract`` for the featurizer class.

    Raises:
        TypeError: If the class contract attribute is not a ``FeatureContract``.
    """
    contract = getattr(cls, "contract", None)
    if contract is None:
        contract = getattr(cls, "output_contract", None)
    if contract is None:
        return FeatureContract(kind=FeatureKind.DENSE)
    if not isinstance(contract, FeatureContract):
        msg = f"Featurizer {cls.__name__!r} contract must be a FeatureContract"
        raise TypeError(msg)
    return contract


def predictor_contracts(cls: type[Any]) -> tuple[FeatureContract, FeatureContract]:
    """Return predictor input contracts, preferring canonical attribute names.

    Args:
        cls: Predictor class registered in the component registry.

    Returns:
        ``(cell_line_contract, drug_contract)`` pair for compatibility checks.

    Raises:
        TypeError: If either contract attribute is not a ``FeatureContract``.
    """
    cell_line = getattr(cls, "cell_line_contract", None)
    if cell_line is None:
        cell_line = getattr(cls, "required_cell_line_contract", None)
    drug = getattr(cls, "drug_contract", None)
    if drug is None:
        drug = getattr(cls, "required_drug_contract", None)
    if cell_line is None:
        cell_line = FeatureContract(kind=FeatureKind.DENSE)
    if drug is None:
        drug = FeatureContract(kind=FeatureKind.DENSE)
    if not isinstance(cell_line, FeatureContract) or not isinstance(drug, FeatureContract):
        msg = f"Predictor {cls.__name__!r} contracts must be FeatureContract instances"
        raise TypeError(msg)
    return cell_line, drug


def contracts_compatible(produced: FeatureContract, required: FeatureContract) -> bool:
    """Return whether *produced* satisfies *required*.

    Args:
        produced: Feature contract emitted by a featurizer.
        required: Feature contract declared by a predictor input slot.

    Returns:
        ``True`` when both contracts share the same ``FeatureKind``.
    """
    return produced.kind == required.kind
