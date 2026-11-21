"""The runtime payload format of a feature block and its config-time spec.

A featurizer declares each block it emits in
:meth:`~drevalpy.components.featurizers.featurizer.Featurizer.output_blocks` as a
:class:`BlockSpec`, and a predictor declares what it reads in ``sample_inputs`` /
``drug_inputs`` (:mod:`drevalpy.components.predictors.inputs`). Config validation accepts
a featurizer for a predictor when the formats of the required blocks match.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureFormat(StrEnum):
    """Runtime payload format of featurizer outputs / predictor inputs.

    * ``NUMERIC_MATRIX``: 2D numeric array with one row per entity.
    * ``GRAPH``: 1D object array of per-entity graph payloads.
    * ``RAGGED_SEQUENCE``: 1D object array of variable-length per-entity arrays (such as
      MolGNet's ``(n_atoms, 768)`` node embeddings where ``n_atoms`` varies per drug).
    """

    NUMERIC_MATRIX = "numeric_matrix"
    GRAPH = "graph"
    RAGGED_SEQUENCE = "ragged_sequence"


@dataclass(frozen=True)
class BlockSpec:
    """Declare a named block emitted by a featurizer or required by a predictor."""

    name: str
    format: FeatureFormat = FeatureFormat.NUMERIC_MATRIX
