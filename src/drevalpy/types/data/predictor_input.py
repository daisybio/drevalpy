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

    ``sample_pair_idx`` and ``drug_pair_idx`` (and, for the early-stopping pairs,
    ``early_stopping_sample_idx`` and ``early_stopping_drug_idx``) are derived from ``sample``
    and ``drug`` on construction. ``pairs.layers`` holds the training-time response layers a predictor declares
    in ``Predictor.response_layers``, one value per pair; it is empty at prediction, where
    ``pairs.response`` is ``NaN``.
    """

    pairs: ResponseTable
    sample: FeaturizerOutput
    drug: FeaturizerOutput | None = None
    early_stopping: ResponseTable | None = None
    training_context: TrainingContext = field(default_factory=TrainingContext)
    sample_pair_idx: np.ndarray = field(init=False)
    drug_pair_idx: np.ndarray | None = field(init=False)
    early_stopping_sample_idx: np.ndarray | None = field(init=False)
    early_stopping_drug_idx: np.ndarray | None = field(init=False)

    def __post_init__(self) -> None:
        """Derive the entity row of every pair and of every early-stopping pair."""
        stopping = self.early_stopping
        object.__setattr__(self, "sample_pair_idx", self.sample.rows_for(self.pairs.sample_ids))
        object.__setattr__(
            self, "drug_pair_idx", None if self.drug is None else self.drug.rows_for(self.pairs.drug_ids)
        )
        object.__setattr__(
            self, "early_stopping_sample_idx", None if stopping is None else self.sample.rows_for(stopping.sample_ids)
        )
        object.__setattr__(
            self,
            "early_stopping_drug_idx",
            None if stopping is None or self.drug is None else self.drug.rows_for(stopping.drug_ids),
        )

    @property
    def sample_ids(self) -> np.ndarray:
        """Sample identifier of each pair."""
        return self.pairs.sample_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """Drug identifier of each pair."""
        return self.pairs.drug_ids

    @property
    def response(self) -> np.ndarray:
        """Response of each pair; ``NaN`` where unknown."""
        return self.pairs.response

    @property
    def response_layers(self) -> Mapping[str, np.ndarray]:
        """Training-time response layers by name, one value per pair."""
        return self.pairs.layers

    def __len__(self) -> int:
        """Return the number of requested sample/drug response pairs.

        :returns: Number of pairs.
        """
        return len(self.pairs)

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
        if not isinstance(selector, slice):
            selector = np.asarray(selector)
            if selector.dtype == bool and selector.shape != (len(self),):
                raise ValueError("subset mask must be a one-dimensional boolean array matching len(self)")
        pairs = self.pairs[selector]
        stopping = self.early_stopping
        if stopping is not None:
            keep = np.isin(stopping.drug_ids, np.unique(pairs.drug_ids))
            stopping = stopping[keep] if keep.any() else None
        return replace(self, pairs=pairs, early_stopping=stopping)
