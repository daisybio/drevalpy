"""
Response transformation utilities.

includes a wrapper class for sklearn transformers
and a custom normalizer that removes cell line and drug effects from response data.
Custom transformers taht require cell lien and drug information can be added here.
"""

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.utils import pipeline_function


class TransformerWrapper(TransformerMixin):
    """
    Wrapper to unify the interface of sklearn-compatible transformers and custom ones.

    Ensures compatibility with transformers such as NaiveMeanEffectsNormalizer
    that require additional keyword arguments.
    """

    def __init__(self, transformer: TransformerMixin) -> None:
        """Initialize the transformer wrapper.

        :param transformer: an sklearn-compatible transformer or a custom transformer
        """
        self.transformer: TransformerMixin = transformer

    def fit(self, X: Sequence[float], y: Optional[Sequence[float]] = None, **kwargs: Any) -> TransformerMixin:
        """
        Fit the wrapped transformer.

        :param X: input array
        :param y: optional targets
        :param kwargs: additional arguments for custom transformers
        :returns: fitted transformer
        """
        if isinstance(self.transformer, NaiveMeanEffectsNormalizer):
            return self.transformer.fit(X, y, **kwargs)
        else:
            return self.transformer.fit(X, y)

    def transform(self, X: Sequence[float], **kwargs: Any) -> np.ndarray:
        """
        Transform input using the wrapped transformer.

        :param X: input array
        :param kwargs: additional arguments for custom transformers
        :returns: transformed array
        """
        if isinstance(self.transformer, NaiveMeanEffectsNormalizer):
            return self.transformer.transform(X, **kwargs)
        else:
            return self.transformer.transform(X)

    def inverse_transform(self, X: Sequence[float], **kwargs: Any) -> Union[np.ndarray, Sequence[float]]:
        """
        Invert transformation if possible.

        :param X: input array
        :param kwargs: additional arguments for custom transformers
        :returns: inverse transformed array
        """
        if hasattr(self.transformer, "inverse_transform"):
            if isinstance(self.transformer, NaiveMeanEffectsNormalizer):
                return self.transformer.inverse_transform(X, **kwargs)
            else:
                return self.transformer.inverse_transform(X)
        return X


@pipeline_function
def get_response_transformation(response_transformation: str) -> Optional[TransformerMixin]:
    """
    Get the sklearn response transformation object of choice.

    Users can choose from "None", "standard", "minmax", "robust", "mean_effects".

    :param response_transformation: response transformation to apply
    :returns: response transformation object
    :raises ValueError: if the response transformation is not recognized
    """
    if response_transformation == "None":
        return None
    if response_transformation == "standard":
        return StandardScaler()
    if response_transformation == "minmax":
        return MinMaxScaler()
    if response_transformation == "robust":
        return RobustScaler()
    if response_transformation == "mean_effects":
        return NaiveMeanEffectsNormalizer()
    raise ValueError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', "
        f"'standard', 'minmax', 'robust', or 'mean_effects'."
    )


class NaiveMeanEffectsNormalizer(TransformerMixin):
    """
    Normalizer that removes additive effects of overall mean, cell line, and drug from drug response data.

    This transformer implements an ANOVA-like normalization:
        response_normalized = response_raw - (overall_mean + cell_line_effect + drug_effect)

    The inverse transform reconstructs the original response values by adding these effects back.
    """

    def __init__(self) -> None:
        """Initialize internal variables for dataset mean, cell line effects, and drug effects."""
        self.dataset_mean: Optional[float] = None
        self.cell_line_effects: dict[str, float] = {}
        self.drug_effects: dict[str, float] = {}

    def fit(
        self,
        X: Sequence[float],
        y: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> "NaiveMeanEffectsNormalizer":
        """
        Compute the dataset mean, and the per-cell-line and per-drug effects.

        :param X: response values
        :param y: unused (for compatibility)
        :param kwargs: must include 'cell_line_ids' and 'drug_ids'
        :returns: self
        :raises ValueError: if required group identifiers are missing
        """
        cell_line_ids = kwargs.get("cell_line_ids")
        drug_ids = kwargs.get("drug_ids")
        if cell_line_ids is None or drug_ids is None:
            raise ValueError("NaiveMeanEffectsNormalizer requires cell_line_ids and drug_ids")

        self.dataset_mean = float(np.mean(X))
        self.cell_line_effects = {
            cl: np.mean(np.array(X)[np.array(cell_line_ids) == cl]) - self.dataset_mean
            for cl in np.unique(cell_line_ids)
        }
        self.drug_effects = {
            d: np.mean(np.array(X)[np.array(drug_ids) == d]) - self.dataset_mean for d in np.unique(drug_ids)
        }
        return self

    def transform(self, X: Sequence[float], *, cell_line_ids: Sequence[str], drug_ids: Sequence[str]) -> np.ndarray:
        """
        Normalize the response values by removing the calculated effects.

        :param X: response values
        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :returns: normalized response values
        :raises ValueError: if the normalizer has not been fitted yet
        """
        if self.dataset_mean is None:
            raise ValueError("Normalizer has not been fitted yet.")

        return np.array(
            [
                X[i] - (self.dataset_mean + self.cell_line_effects.get(cl, 0) + self.drug_effects.get(d, 0))
                for i, (cl, d) in enumerate(zip(cell_line_ids, drug_ids))
            ]
        )

    def inverse_transform(
        self, X_norm: Sequence[float], *, cell_line_ids: Sequence[str], drug_ids: Sequence[str]
    ) -> np.ndarray:
        """
        Reconstruct the original response values by adding the effects back.

        :param X_norm: normalized response values
        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :returns: reconstructed original response values
        :raises ValueError: if the normalizer has not been fitted yet
        """
        if self.dataset_mean is None:
            raise ValueError("Normalizer has not been fitted yet.")

        return np.array(
            [
                X_norm[i] + self.dataset_mean + self.cell_line_effects.get(cl, 0) + self.drug_effects.get(d, 0)
                for i, (cl, d) in enumerate(zip(cell_line_ids, drug_ids))
            ]
        )
