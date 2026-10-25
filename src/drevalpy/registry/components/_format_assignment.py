"""Feature format resolution for component registration.

Deciding *which* format a component ends up with - the one passed to the registration
decorator or the one declared in the class body - is a pure function of the class and
the decorator argument. It touches no registry state, which is what made
``ComponentRegistry`` incohesive while it lived there, and it is shared by the featurizer
registry (``output_format``) and the predictor registry (``sample_input_format`` and
``drug_input_format``).
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.feature_format import FeatureFormat, check_format
from drevalpy.log import get_logger

logger = get_logger(__name__)


def assign_format(cls: type[Any], attr: str, feature_format: FeatureFormat | None) -> None:
    """Set *attr* on *cls* to the format that wins, then check it.

    A class-body declaration is used whenever the registration decorator was given none.
    When both are present the decorator wins, since it is the more specific of the two.

    :param cls: Class being registered.
    :param attr: Attribute name: ``output_format``, ``sample_input_format`` or ``drug_input_format``.
    :param feature_format: Format passed to the decorator, or ``None`` to use the class declaration.
    """
    if feature_format is not None:
        if attr in cls.__dict__:
            logger.debug("%s: decorator %s overrides the class-body declaration", cls.__name__, attr)
        setattr(cls, attr, feature_format)
    check_format(cls, attr)
