"""The record of a randomization test behind :meth:`Dataset.with_randomized_views`.

A randomization test leaves the data of the dataset as they are. ``with_randomized_views`` records
which views to randomize, how and with which seed under ``uns["randomization_provenance"]``, and a
model trained on the dataset randomizes the output of each featurizer that reads one of those views
(:mod:`drevalpy.components.featurizers._randomized`). The featurizer runs with its own settings, so a
tuned ``fingerprints`` with ``n_bits=512`` randomizes its 512-bit output. Randomizing the stored view
instead would leave it at the settings it was precomputed with, or be bypassed when the featurizer
computes its input from the SMILES, as it does for a drug without a stored graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from drevalpy.types.enums import RandomizationType


@dataclass(frozen=True)
class ViewRandomization:
    """The views a randomization test randomizes, how, and the seed every draw derives from."""

    #: Names of the views whose featurizer output is randomized.
    views: frozenset[str]
    #: How the output is randomized.
    randomization_type: RandomizationType
    #: Seed of the draws.
    seed: int

    @classmethod
    def from_provenance(cls, provenance: Mapping[str, Any]) -> ViewRandomization:
        """Read the record from a dataset's ``uns["randomization_provenance"]``.

        :param provenance: The stored mapping, with the keys :meth:`provenance` writes.
        :returns: The record.
        """
        return cls(
            frozenset(provenance["views"]),
            RandomizationType(provenance["type"]),
            int(provenance["effective_seed"]),
        )

    def provenance(self) -> dict[str, Any]:
        """Return the record as it is stored in ``uns["randomization_provenance"]``.

        :returns: ``effective_seed``, ``type`` and the sorted ``views``.
        """
        return {"effective_seed": self.seed, "type": self.randomization_type.value, "views": sorted(self.views)}
