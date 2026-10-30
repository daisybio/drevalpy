"""The runtime payload format of a feature block.

A featurizer declares the format of each block it emits in
:meth:`~drevalpy.components.featurizers.featurizer.Featurizer.output_blocks`, and a
predictor the format of each block it reads in its ``sample_inputs`` / ``drug_inputs``
(:mod:`drevalpy.components.predictors.inputs`). Config validation accepts a featurizer
for a predictor when the formats of the required blocks are equal.
"""

from __future__ import annotations

from enum import StrEnum


class FeatureFormat(StrEnum):
    """Runtime payload format of featurizer outputs / predictor inputs."""

    NUMERIC_MATRIX = "numeric_matrix"
    GRAPH = "graph"
    RAGGED_SEQUENCE = "ragged_sequence"
