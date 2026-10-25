"""One run identity shared by experiment execution and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    """A named model/fold plus all stress-test dimensions and recorded seeds."""

    model_name: str
    dataset_name: str
    split_mode: str
    fold_index: int
    fold_id: str = ""
    randomization: tuple[str, str] | None = None
    robustness_repeat: int | None = None
    provenance: str = "{}"

    @property
    def fold_key(self) -> tuple[str, str, str, str | int]:
        """Identity shared by ordinary and stress-test runs of one fold."""
        return (
            self.dataset_name,
            self.split_mode,
            "id" if self.fold_id else "index",
            self.fold_id if self.fold_id else self.fold_index,
        )

    @property
    def run_key(self) -> tuple:
        """Include stress-test settings and seeds without losing ordinary grouping."""
        return (self.model_name, *self.fold_key, self.randomization, self.robustness_repeat, self.provenance)

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
        """Build an identity from persisted result or run metadata.

        :param model_name: Named recipe.
        :param dataset_name: Dataset identity.
        :param split_mode: Split protocol.
        :param fold_index: Fold ordinal.
        :param fold_id: Stable fold identifier, when available.
        :param randomization: Optional ``(mode, view)`` stress test.
        :param metadata: Recorded fold and run provenance.
        :returns: Immutable run identity.
        """
        identity = {key: metadata[key] for key in ("randomization_provenance", "execution") if key in metadata}
        return cls(
            model_name,
            dataset_name,
            split_mode,
            fold_index,
            fold_id,
            randomization,
            metadata.get("robustness_repeat"),
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        )
