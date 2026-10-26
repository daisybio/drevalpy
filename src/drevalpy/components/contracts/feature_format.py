"""The runtime payload format a featurizer emits and a predictor consumes.

A featurizer declares ``output_format``; a predictor declares ``sample_input_format`` and
``drug_input_format``. Each may be set on the class body or passed to the registration
decorator, which wins when both are given. A predictor accepts a featurizer when the two
formats are equal.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class FeatureFormat(StrEnum):
    """Runtime payload format of featurizer outputs / predictor inputs."""

    NUMERIC_MATRIX = "numeric_matrix"
    GRAPH = "graph"
    RAGGED_SEQUENCE = "ragged_sequence"


def check_format(cls: type[Any], attr: str) -> FeatureFormat:
    """Return the ``FeatureFormat`` that *cls* declares under *attr*.

    :param cls: Featurizer or predictor class.
    :param attr: ``"output_format"``, ``"sample_input_format"`` or ``"drug_input_format"``.

    :returns: The declared format.

    :raises TypeError: If *cls* does not declare *attr*, or it is not a ``FeatureFormat``.
    """
    value = getattr(cls, attr, None)
    if not isinstance(value, FeatureFormat):
        msg = (
            f"{cls.__name__}.{attr} must be a FeatureFormat, got {value!r}; "
            f"set it on the class body or pass {attr}= to the registration decorator"
        )
        raise TypeError(msg)
    return value
