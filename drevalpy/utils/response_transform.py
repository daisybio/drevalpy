"""Unfitted sklearn target-transform prototypes for model configuration.

``sklearn`` is imported inside the lookup function rather than at module scope:
``drevalpy.models.drp_model`` reaches this module on the registration path of
``import drevalpy``, and importing ``sklearn.base`` alone costs ~0.35s because it
pulls in ``scipy.stats``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def get_response_transformation(
    response_transformation: str | None,
) -> TransformerMixin | None:
    """Return the sklearn response transformer for a pipeline option.

    :param response_transformation: One of ``"None"``, ``"standard"``, ``"minmax"``, or ``"robust"``.

    :returns: Fitted-ready sklearn transformer, or ``None`` for no transformation.

    :raises ValueError: If *response_transformation* is not recognized.

    :param response_transformation: response transformation.
    :returns: Result of the operation.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    if (response_transformation == "None") or (response_transformation is None):
        return None
    if response_transformation == "standard":
        return StandardScaler()
    if response_transformation == "minmax":
        return MinMaxScaler()
    if response_transformation == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', 'standard', 'minmax', 'robust'"
    )
