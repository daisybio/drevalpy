"""Pick a response transformation by its pipeline option name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.errors import UnknownNameError

if TYPE_CHECKING:
    from ._protocol import ResponseTransformer


def get_response_transformation(response_transformation: str | None) -> ResponseTransformer | None:
    """Return the response transformation for a pipeline option.

    ``"drug_mean"`` subtracts the per-drug mean of the training scope. ``"drug_tissue_mean"`` does the
    same per drug and tissue, falling back to the per-drug mean for a (drug, tissue) combination the
    training scope did not contain. Both add the mean back to the predictions.

    :param response_transformation: ``None`` or ``"none"`` (in any letter case) for no transformation, or one of
        ``"standard"``, ``"minmax"``, ``"robust"``, ``"drug_mean"`` and ``"drug_tissue_mean"``.
    :returns: A new, unfitted transformer, or ``None`` for no transformation.
    :raises UnknownNameError: If *response_transformation* is not recognized.
    """
    if response_transformation is None or response_transformation.lower() == "none":
        return None
    if response_transformation in ("drug_mean", "drug_tissue_mean"):
        from ._group_mean import GroupMeanCenterer

        return GroupMeanCenterer(("drug",) if response_transformation == "drug_mean" else ("drug", "tissue"))
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if response_transformation in scalers:
        return scalers[response_transformation]()
    raise UnknownNameError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', 'standard', 'minmax', "
        "'robust', 'drug_mean', 'drug_tissue_mean'"
    )
