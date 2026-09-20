"""One execution identity shared by experiment execution and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    """A named model/fold plus all diagnostic dimensions and recorded seeds."""

    model_name: str
    dataset_name: str
    split_mode: str
    fold_index: int
    fold_id: str = ""
    randomization: tuple[str, str] | None = None
    robustness_trial: int | None = None
    provenance: str = "{}"

    @property
    def fold_key(self) -> tuple[str, str, str, str | int]:
        """Identity shared by ordinary and diagnostic executions of one fold."""
        return (
            self.dataset_name,
            self.split_mode,
            "id" if self.fold_id else "index",
            self.fold_id if self.fold_id else self.fold_index,
        )

    @property
    def execution_key(self) -> tuple:
        """Include diagnostic settings and seeds without losing ordinary grouping."""
        return (self.model_name, *self.fold_key, self.randomization, self.robustness_trial, self.provenance)

    @classmethod
    def from_metadata(
        cls,
        model_name: str,
        dataset_name: str,
        split_mode: str,
        fold_index: int,
        fold_id: str,
        randomization: tuple[str, str] | None,
        metadata: dict[str, Any],
    ) -> RunIdentity:
        """Build an identity from persisted result or execution metadata.

        :param model_name: Named recipe.
        :param dataset_name: Dataset identity.
        :param split_mode: Split protocol.
        :param fold_index: Fold ordinal.
        :param fold_id: Stable fold identifier, when available.
        :param randomization: Optional diagnostic mode/view.
        :param metadata: Recorded fold and execution provenance.
        :returns: Immutable execution identity.
        """
        identity = {key: metadata[key] for key in ("randomization_provenance", "execution") if key in metadata}
        return cls(
            model_name,
            dataset_name,
            split_mode,
            fold_index,
            fold_id,
            randomization,
            metadata.get("robustness_trial"),
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        )
