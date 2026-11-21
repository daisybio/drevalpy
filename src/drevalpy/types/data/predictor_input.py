"""Canonical predictor input container for component-based models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.feature_block import FeaturizerOutput
from drevalpy.types.data.response_table import ResponseTable


@dataclass(frozen=True)
class PredictorInput:
    """Response pairs indexed into canonical per-side featurizer results.

    ``sample_pair_idx`` and ``drug_pair_idx`` are derived from ``sample`` and ``drug`` on
    construction. ``response_layers`` holds the training-time response layers a predictor
    declares in ``Predictor.response_layers``, one value per pair; it is empty at prediction.
    """

    sample_ids: np.ndarray
    drug_ids: np.ndarray
    response: np.ndarray | None
    sample: FeaturizerOutput
    drug: FeaturizerOutput | None = None
    early_stopping: ResponseTable | None = None
    training_context: TrainingContext = field(default_factory=TrainingContext)
    response_layers: Mapping[str, np.ndarray] = field(default_factory=dict)
    sample_pair_idx: np.ndarray = field(init=False)
    drug_pair_idx: np.ndarray | None = field(init=False)

    def __post_init__(self) -> None:
        """Validate pair lengths and derive the entity row of every pair.

        :raises ValueError: If a pair array is not one-dimensional with one value per pair.
        """
        sample_ids, drug_ids = np.asarray(self.sample_ids), np.asarray(self.drug_ids)
        if sample_ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        n = len(sample_ids)
        pair_arrays = {"drug_ids": drug_ids, "response": self.response}
        pair_arrays |= {f"response layer {name!r}": values for name, values in self.response_layers.items()}
        for name, values in pair_arrays.items():
            if values is not None and (np.ndim(values) != 1 or len(values) != n):
                raise ValueError(f"{name} must be one-dimensional with {n} entries")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "drug_ids", drug_ids)
        if self.response is not None:
            object.__setattr__(self, "response", np.asarray(self.response, dtype=np.float64))
        object.__setattr__(self, "sample_pair_idx", self.sample.rows_for(self.sample_ids))
        object.__setattr__(self, "drug_pair_idx", None if self.drug is None else self.drug.rows_for(self.drug_ids))

    def __len__(self) -> int:
        """Return the number of requested sample/drug response pairs.

        :returns: Length of ``sample_ids``.
        """
        return len(self.sample_ids)

    @classmethod
    def from_response(
        cls,
        response: ResponseTable,
        *,
        sample: FeaturizerOutput,
        drug: FeaturizerOutput | None = None,
        early_stopping: ResponseTable | None = None,
        training_context: TrainingContext | None = None,
    ) -> PredictorInput:
        """Index response pairs into already computed entity-level results.

        :param response: Requested pairs and optional targets; its ``layers`` become ``response_layers``.
        :param sample: Canonical sample features.
        :param drug: Optional canonical drug features.
        :param early_stopping: Optional disjoint stopping pairs.
        :param training_context: Runtime checkpoint and logging context.
        :returns: Aligned predictor input in response order.
        """
        return cls(
            response.sample_ids,
            response.drug_ids,
            response.response,
            sample,
            drug,
            early_stopping,
            training_context or TrainingContext(),
            response.layers,
        )

    def to_feature_matrix(self) -> np.ndarray:
        """Return the dense design matrix in the input's requested pair order.

        Raises ``ValueError`` (via :attr:`FeaturizerOutput.matrix`) if a numeric feature
        block is not a two-dimensional numeric array.

        :returns: Sample and drug feature matrices side by side, one row per pair.
        """
        arrays = []
        for table, indices in ((self.sample, self.sample_pair_idx), (self.drug, self.drug_pair_idx)):
            if table is None or indices is None:
                continue
            matrix = table.matrix
            if matrix.shape[1] == 0:
                continue
            arrays.append(matrix[indices])
        if not arrays:
            return np.empty((len(self), 0), dtype=np.float32)
        return np.concatenate(arrays, axis=1)

    def __getitem__(self, selector: np.ndarray | slice) -> PredictorInput:
        """Select pairs by boolean/integer mask or slice, sharing entity payloads and restricting stopping drugs.

        :param selector: Boolean mask, integer index array, or slice over the pairs.
        :returns: Independent pair selection over the same canonical features.
        :raises ValueError: If a boolean mask does not match ``len(self)``.
        """
        index: np.ndarray | slice
        if isinstance(selector, slice):
            index = selector
        else:
            index = np.asarray(selector)
            if index.dtype == bool and (index.ndim != 1 or len(index) != len(self)):
                raise ValueError("subset mask must be a one-dimensional boolean array matching len(self)")
        selected_drugs = self.drug_ids[index]
        stopping = self.early_stopping
        if stopping is not None:
            keep = np.isin(stopping.drug_ids, np.unique(selected_drugs))
            stopping = stopping[keep] if keep.any() else None
        return replace(
            self,
            sample_ids=self.sample_ids[index],
            drug_ids=selected_drugs,
            response=None if self.response is None else self.response[index],
            early_stopping=stopping,
            response_layers={name: values[index] for name, values in self.response_layers.items()},
        )
