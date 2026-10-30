"""Morgan fingerprint drug featurizer: the stored view when the dataset has one, else computed with RDKit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

_BLOCK = "fingerprints"


@register(
    "fingerprints",
    description="Morgan fingerprints loaded from pre-computed view or computed on the fly via rdkit.",
)
class FingerprintsFeaturizer(Featurizer):
    """Morgan fingerprints featurizer with on-the-fly fallback."""

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    storage_key: ClassVar[str | None] = "morgan_fingerprint"
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
        self._view = view
        self._radius = int(radius)
        self._n_bits = int(n_bits)
        self._use_chirality = bool(use_chirality)
        self._use_counts = bool(use_counts)
        self._view_names: tuple[str, ...] | None = None

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

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single ``fingerprints`` block, whatever view it is read from.

        :param config: Featurizer config; unused.
        :returns: One numeric block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match a stored variant only when it was computed with the same non-default settings.

        :returns: The hyperparameters that differ from their defaults; empty for the default config.
        """
        settings = {
            "radius": self._radius,
            "n_bits": self._n_bits,
            "use_chirality": self._use_chirality,
            "use_counts": self._use_counts,
        }
        defaults = self.default_hyperparameters()
        return {name: value for name, value in settings.items() if value != defaults[name]}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap a stored matrix as the block, named by the view's features.

        :param source: Drug feature source.
        :param ids: Drug IDs, one per row of *matrix*.
        :param matrix: Stored matrix.
        :returns: Result with the single block.
        """
        return dense_result(ids, _BLOCK, matrix, source.get_feature_names(self._view))

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Record the view's feature names; nothing is computed.

        :param source: Drug feature source.
        :param train_ids: Usable training drug IDs; unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        self._view_names = source.get_feature_names(self._view)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Read the stored fingerprints when the source holds the view, else compute them from the SMILES.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with the ``fingerprints`` block.
        """
        if source.has_view(self._view):
            matrix = view_matrix(source, self._view, ids, self._view_names)
            return dense_result(ids, _BLOCK, matrix, self._view_names)
        return dense_result(ids, _BLOCK, self._from_smiles(source, ids), None)

    def _from_smiles(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Compute Morgan fingerprints from SMILES in one RDKit pass.

        :param source: Drug feature source.
        :param ids: Drug identifiers.
        :returns: Fingerprint matrix of shape ``(n_drugs, n_bits)``; NaN rows for unusable SMILES.
        :raises ValueError: If the source has no SMILES.
        :raises ImportError: If rdkit is not installed.
        """
        smiles = get_smiles_for_entities(source, ids)
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
