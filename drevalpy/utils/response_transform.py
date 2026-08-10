"""Sklearn response-value transformations for the evaluation pipeline."""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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
