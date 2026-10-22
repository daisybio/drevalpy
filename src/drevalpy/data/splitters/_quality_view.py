"""The dataset view :func:`drevalpy.data.split` hands to a splitter.

A splitter only reads ``isfinite(response_matrix)``. Applying the curve-quality
rule *before* it, as NaN on the failed pairs, keeps the splitter protocol
unchanged: plugin splitters need no knowledge of :class:`CurveQuality`, and a
dataset without quality layers can be split once the rule is
:meth:`~drevalpy.data.quality.CurveQuality.off`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality
    from drevalpy.types import MuDataLike


class QualityMasked:
    """A :class:`~drevalpy.types.data.mudatalike.MuDataLike` whose failed pairs read as NaN.

    Everything but ``response_matrix`` is delegated to the wrapped dataset,
    including the response layers, which keep their values for the failed pairs.
    """

    def __init__(self, dataset: MuDataLike, rule: CurveQuality) -> None:
        """Evaluate *rule* on *dataset* once and blank the pairs that fail it.

        :param dataset: Dataset to wrap; it is not modified.
        :param rule: Curve-quality rule to apply.
        :raises KeyError: If *rule* needs a response layer the dataset lacks.
        """
        self._dataset = dataset
        response = np.array(dataset.response_matrix, copy=True)
        response[~rule.mask(dataset)] = np.nan
        self._response = response

    @property
    def sample_ids(self) -> np.ndarray:
        """Row identifiers of the response matrix."""
        return self._dataset.sample_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers of the response matrix."""
        return self._dataset.drug_ids

    @property
    def response_matrix(self) -> np.ndarray:
        """Sample-by-drug response with NaN for unmeasured pairs and pairs failing the rule."""
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return the tissue label of each sample in *ids*."""
        return self._dataset.get_tissue(ids)

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Return one per-sample metadata column for the samples in *ids*."""
        return self._dataset.sample_column(column, ids)

    def response_layer_names(self) -> list[str]:
        """Names of the response layers of the wrapped dataset."""
        return self._dataset.response_layer_names()

    def get_response_layer(self, name: str) -> np.ndarray:
        """Return a response layer of the wrapped dataset, unmasked."""
        return self._dataset.get_response_layer(name)

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attribute to the wrapped dataset."""
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)
