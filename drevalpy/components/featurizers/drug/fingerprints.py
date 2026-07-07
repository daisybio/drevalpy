"""Morgan fingerprint drug featurizer."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "fingerprints",
    description="Precomputed Morgan fingerprints loaded from the fingerprints view.",
    category="general_purpose",
    contract=FeatureKind.DENSE,
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Alias for the standard fingerprints view."""

    def __init__(self, *, view: str = "fingerprints") -> None:
        super().__init__(view=view)
