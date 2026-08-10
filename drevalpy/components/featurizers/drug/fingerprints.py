"""Morgan fingerprint drug featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger

_logger = get_logger(__name__)


@register_drug_featurizer(
    "fingerprints",
    description="Morgan fingerprints loaded from pre-computed view or computed on the fly via rdkit.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Morgan fingerprints featurizer with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "morgan_fingerprint"
    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = True

    def __init__(
        self,
        *,
        view: str = "morgan_fingerprint",
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_counts: bool = False,
    ) -> None:
        """Initialize instance state.

        :param view: view.
        :param radius: Morgan fingerprint radius (neighborhood extent).
        :param n_bits: Fingerprint bit length.
        :param use_chirality: Whether to include stereochemistry information.
        :param use_counts: Whether to use count-based (True) or binary (False) fingerprints.
        """
        super().__init__(view=view)
        self._radius = int(radius)
        self._n_bits = int(n_bits)
        self._use_chirality = bool(use_chirality)
        self._use_counts = bool(use_counts)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "radius": {"type": "int", "low": 1, "high": 3, "default": 2},
            "n_bits": {"type": "pow2", "low": 9, "high": 12, "default": 2048},
            "use_chirality": {"type": "categorical", "choices": [True, False], "default": False},
            "use_counts": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute Morgan fingerprints from SMILES.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Fingerprint matrix of shape (n_drugs, n_bits).
        """
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = f"Cannot obtain {self.storage_key}: no SMILES available."
            raise ValueError(msg)

        try:
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError as err:
            msg = "rdkit is required for on-the-fly fingerprint computation: pip install rdkit"
            raise ImportError(msg) from err

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius, fpSize=self._n_bits, includeChirality=self._use_chirality
        )
        results = np.zeros((len(entity_ids), self._n_bits), dtype=np.float32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles.get(drug_id)
            results[i] = _fingerprint_for_smiles(smi, generator, self._n_bits, self._use_counts)
        return results

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Mapping with one numeric block.
        """
        return {
            "fingerprints": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }


def _fingerprint_for_smiles(smi, generator, n_bits: int, use_counts: bool) -> np.ndarray:
    """Compute a fingerprint for one SMILES string."""
    from rdkit import Chem

    if not smi or not isinstance(smi, str):
        return np.full(n_bits, np.nan, dtype=np.float32)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.full(n_bits, np.nan, dtype=np.float32)
    if use_counts:
        return generator.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    return generator.GetFingerprintAsNumPy(mol).astype(np.float32)
