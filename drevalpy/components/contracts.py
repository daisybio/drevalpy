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

    This class should store all properties that are required to check if a featurizer is compatible with a predictor.
    Currently, we only store the ``FeatureFormat``.
    In future, we might want to store additional properties, like the type of graph.
    """

    format: FeatureFormat


def normalize_feature_contract(contract: FeatureContract | FeatureFormat) -> FeatureContract:
    """Return a ``FeatureContract`` from a contract object or format shorthand.

    As our FeatureContracts currently only store the ``FeatureFormat``, the
    featurizer and predictor decorators allow providing either a full
    ``FeatureContract`` instance or just the ``FeatureFormat`` enum member.
    This function makes sure that we always return a ``FeatureContract``
    instance, even if just a ``FeatureFormat`` is provided.

    :param contract: A ``FeatureContract`` instance or ``FeatureFormat`` enum member.

    :returns: Normalized ``FeatureContract`` instance.

    :raises TypeError: If *contract* is neither ``FeatureContract`` nor ``FeatureFormat``.
    """
    if isinstance(contract, FeatureContract):
        return contract
    if isinstance(contract, FeatureFormat):
        return FeatureContract(format=contract)
    msg = f"Expected FeatureContract or FeatureFormat, got {type(contract).__name__}"
    raise TypeError(msg)


def featurizer_contract(cls: type[Any]) -> FeatureContract:
    """Return the featurizer contract from ``contract``.

    :param cls: Featurizer class registered in the component registry.

    :returns: Resolved ``FeatureContract`` for the featurizer class.

    :raises TypeError: If the class has no ``contract``, or it is not a ``FeatureContract``.
    """
    contract = getattr(cls, "contract", None)
    if contract is None:
        raise TypeError(f"Featurizer {cls.__name__!r} must define a contract")
    if not isinstance(contract, FeatureContract):
        raise TypeError(f"Featurizer {cls.__name__!r} contract must be a FeatureContract")
    return contract


def predictor_contracts(cls: type[Any]) -> tuple[FeatureContract, FeatureContract]:
    """Return predictor input contracts for cell-line and drug sides.

    :param cls: Predictor class registered in the component registry.

    :returns: ``(cell_line_contract, drug_contract)`` pair for compatibility checks.

    :raises TypeError: If either contract is missing or not a ``FeatureContract``.
    """
    cell_line = getattr(cls, "cell_line_contract", None)
    drug = getattr(cls, "drug_contract", None)
    if cell_line is None or drug is None:
        msg = f"Predictor {cls.__name__!r} must define both " "cell_line_contract and drug_contract"
        raise TypeError(msg)
    if not isinstance(cell_line, FeatureContract) or not isinstance(drug, FeatureContract):
        msg = f"Predictor {cls.__name__!r} contracts must be FeatureContract instances"
        raise TypeError(msg)
    return cell_line, drug


def contracts_compatible(produced: FeatureContract, required: FeatureContract) -> bool:
    """Return whether *produced* satisfies *required*.

    Currently, this only checks if the ``FeatureFormat`` is the same.
    In future, we might want to check additional properties, like the type of graph.
    If we do, we need to ensure that the ``FeatureContract`` class is extended to store the additional properties.

    :param produced: Feature contract emitted by a featurizer.
    :param required: Feature contract declared by a predictor input slot.

    :returns: ``True`` when both contracts share the same ``FeatureFormat``.
    """
    return produced.format == required.format
