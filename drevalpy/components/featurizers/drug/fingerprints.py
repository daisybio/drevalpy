"""Morgan fingerprint drug featurizer."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "fingerprints",
    description="Precomputed Morgan fingerprints loaded from the fingerprints view.",
    category="general_purpose",
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Alias for the standard fingerprints view."""

    def __init__(self, *, view: str = "fingerprints", n_bits: int = 128) -> None:
        super().__init__(view=view)
        self._n_bits = int(n_bits)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_bits": {
                "type": "categorical",
                "choices": [128, 256, 512],
                "default": 128,
            },
        }
