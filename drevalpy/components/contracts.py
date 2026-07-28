"""Feature format and contract objects for component compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class FeatureFormat(StrEnum):
    """Runtime payload format of featurizer outputs / predictor inputs."""

    NUMERIC_MATRIX = "numeric_matrix"
    GRAPH = "graph"
    RAGGED_SEQUENCE = "ragged_sequence"


@dataclass(frozen=True)
class FeatureContract:
    """Structured description of a feature representation.

    Compatibility checks the ``FeatureFormat``. Graph and ragged payloads may
    carry additional validation elsewhere (container type, required attributes).
    """

    format: FeatureFormat


def normalize_feature_contract(contract: FeatureContract | FeatureFormat) -> FeatureContract:
    """Return a ``FeatureContract`` from a contract object or format shorthand.

    Args:
        contract: A ``FeatureContract`` instance or ``FeatureFormat`` enum member.

    Returns:
        Normalized ``FeatureContract`` instance.

    Raises:
        TypeError: If *contract* is neither ``FeatureContract`` nor ``FeatureFormat``.
    """
    if isinstance(contract, FeatureContract):
        return contract
    if isinstance(contract, FeatureFormat):
        return FeatureContract(format=contract)
    msg = f"Expected FeatureContract or FeatureFormat, got {type(contract).__name__}"
    raise TypeError(msg)


def featurizer_contract(cls: type[Any]) -> FeatureContract:
    """Return the featurizer contract from ``contract``.

    Args:
        cls: Featurizer class registered in the component registry.

    Returns:
        Resolved ``FeatureContract`` for the featurizer class.

    Raises:
        TypeError: If the class contract attribute is not a ``FeatureContract``.
    """
    contract = getattr(cls, "contract", None)
    if contract is None:
        return FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    if not isinstance(contract, FeatureContract):
        msg = f"Featurizer {cls.__name__!r} contract must be a FeatureContract"
        raise TypeError(msg)
    return contract


def predictor_contracts(cls: type[Any]) -> tuple[FeatureContract, FeatureContract]:
    """Return predictor input contracts for cell-line and drug sides.

    Args:
        cls: Predictor class registered in the component registry.

    Returns:
        ``(cell_line_contract, drug_contract)`` pair for compatibility checks.

    Raises:
        TypeError: If either contract attribute is not a ``FeatureContract``.
    """
    cell_line = getattr(cls, "cell_line_contract", None)
    drug = getattr(cls, "drug_contract", None)
    if cell_line is None:
        cell_line = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    if drug is None:
        drug = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
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
        ``True`` when both contracts share the same ``FeatureFormat``.
    """
    return produced.format == required.format
