"""Morgan fingerprint drug featurizer: the precomputed view when it matches the settings, else computed with RDKit."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.featurizers._dense import PrecomputableFeaturizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.feature_source import FeatureSource, Side


@register(
    "fingerprints",
    description="Morgan fingerprints loaded from pre-computed view or computed on the fly via rdkit.",
)
class FingerprintsFeaturizer(PrecomputableFeaturizer):
    """Morgan fingerprints featurizer with on-the-fly fallback."""

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    block: ClassVar[str | None] = "fingerprints"
    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(
        self,
        *,
        view: str = "morgan_fingerprint",
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_counts: bool = False,
    ) -> None:
        """Configure the Morgan fingerprint computed when the view is absent.

        :param view: Name of the view holding pre-computed fingerprints.
        :param radius: Morgan fingerprint radius (neighborhood extent).
        :param n_bits: Fingerprint bit length.
        :param use_chirality: Whether to include stereochemistry information.
        :param use_counts: Whether to use count-based (True) or binary (False) fingerprints.
        """
        self._radius = int(radius)
        self._n_bits = int(n_bits)
        self._use_chirality = bool(use_chirality)
        self._use_counts = bool(use_counts)
        super().__init__(
            view,
            {
                "radius": self._radius,
                "n_bits": self._n_bits,
                "use_chirality": self._use_chirality,
                "use_counts": self._use_counts,
            },
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "radius": {"type": "int", "low": 1, "high": 3, "default": 2},
            "n_bits": {"type": "pow2", "low": 9, "high": 11, "default": 2048},
            "use_chirality": {"type": "categorical", "choices": [True, False], "default": False},
            "use_counts": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def compute(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Compute Morgan fingerprints from SMILES in one RDKit pass.

        :param source: Drug feature source.
        :param ids: Drug identifiers.
        :returns: Fingerprint matrix of shape ``(n_drugs, n_bits)``; NaN rows for unusable SMILES.
        :raises ValueError: If the source has no SMILES.
        :raises ImportError: If rdkit is not installed.
        """
        smiles = source.smiles(ids)
        if smiles is None:
            msg = f"Cannot compute {self._view}: no SMILES available."
            raise ValueError(msg)

        try:
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError as err:
            msg = "rdkit is required for on-the-fly fingerprint computation: pip install rdkit"
            raise ImportError(msg) from err

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self._radius, fpSize=self._n_bits, includeChirality=self._use_chirality
        )
        results = np.zeros((len(ids), self._n_bits), dtype=np.float32)
        for i, drug_id in enumerate(ids):
            results[i] = _fingerprint_for_smiles(smiles.get(drug_id), generator, self._n_bits, self._use_counts)
        return results


def _fingerprint_for_smiles(smi: object, generator: Any, n_bits: int, use_counts: bool) -> np.ndarray:
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
