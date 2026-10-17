"""Contract resolution for component registration.

Deciding *which* contract a component ends up with - the one passed to the
registration decorator or the one declared in the class body - is a pure function
of the class and the decorator argument. It touches no registry state, which is
what made ``ComponentRegistry`` incohesive while it lived there, and it is shared
verbatim by the featurizer registry (one ``contract``) and the predictor registry
(a ``cell_line_contract`` and a ``drug_contract``).
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.contracts import FeatureContract, normalize_feature_contract
from drevalpy.log import get_logger

logger = get_logger(__name__)


def assign_contract(cls: type[Any], attr_name: str, contract: FeatureContract | None) -> None:
    """Set *attr_name* on *cls* to the contract that wins.

    A class-body declaration is a valid way to state a contract: it is used
    whenever the registration decorator was given none. When both are present the
    decorator wins, since it is the more specific of the two.

    :param cls: Class being registered.
    :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
    :param contract: Already-normalized feature contract from the decorator, or
        ``None`` to fall back to the class declaration.
    :raises ValueError: If neither the decorator nor the class declares *attr_name*.
    """
    if contract is None:
        contract = declared_contract(cls, attr_name)
    elif attr_name in cls.__dict__:
        logger.debug(
            "%s: decorator %s overrides the class-body declaration",
            cls.__name__,
            attr_name,
        )
    setattr(cls, attr_name, contract)


def declared_contract(cls: type[Any], attr_name: str) -> FeatureContract:
    """Return the contract *cls* declares under *attr_name*.

    :param cls: Class being registered.
    :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
    :returns: Normalized contract taken from the class declaration.
    :raises ValueError: If the class declares no usable contract.
    """
    declared = getattr(cls, attr_name, None)
    if declared is None:
        msg = (
            f"{cls.__name__}: no {attr_name} declared; pass {attr_name}= to the "
            f"registration decorator or set it on the class body"
        )
        raise ValueError(msg)
    try:
        return normalize_feature_contract(declared)
    except TypeError as exc:
        msg = f"{cls.__name__}: class-body {attr_name} is invalid: {exc}"
        raise ValueError(msg) from exc
