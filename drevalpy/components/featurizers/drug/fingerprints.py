"""Morgan fingerprint drug featurizer."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "fingerprints",
    description="Precomputed Morgan fingerprints loaded from the fingerprints view.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Alias for the standard fingerprints view."""

    def __init__(self, *, view: str = "fingerprints") -> None:
        """Initialize instance state.

        :param view: view.
        """
        super().__init__(view=view)
