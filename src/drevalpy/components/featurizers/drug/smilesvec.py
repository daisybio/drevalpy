"""SMILESVec drug featurizer with auto-download of pre-trained Word2Vec model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.data.artifacts import get_artifact
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

_SMILESVEC_MODEL_FILE = "drug.pubchem.canon.l8.ws20.txt"
_BLOCK = "smilesvec"


@register(
    "smilesvec",
    description="SMILESVec drug embeddings computed on the fly via a pre-trained Word2Vec model.",
)
class SmilesVecDrugFeaturizer(Featurizer):
    """SMILESVec featurizer using Word2Vec k-mer embeddings from PubChem corpus."""

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    storage_key: ClassVar[str | None] = "smilesvec"
    input_views: ClassVar[tuple[str, ...]] = ("smilesvec",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "smilesvec", k: int = 8) -> None:
        """Configure the k-mer length of the on-the-fly embedding.

        :param view: Name of the view holding pre-computed embeddings.
        :param k: Length of SMILES subsequences (chemical words).
        """
        self._view = view
        self._k = int(k)
        self._view_names: tuple[str, ...] | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "k": {"type": "categorical", "choices": [4, 6, 8, 10, 12], "default": 8},
        }

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single ``smilesvec`` block, whatever view it is read from.

        :param config: Featurizer config; unused.
        :returns: One numeric block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match a stored variant only when it was computed with the same non-default settings.

        :returns: The hyperparameters that differ from their defaults; empty for the default config.
        """
        settings = {"k": self._k}
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
        """Read the stored embeddings when the source holds the view, else compute them from the SMILES.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with the ``smilesvec`` block.
        """
        if source.has_view(self._view):
            matrix = view_matrix(source, self._view, ids, self._view_names)
            return dense_result(ids, _BLOCK, matrix, self._view_names)
        return dense_result(ids, _BLOCK, self._from_smiles(source, ids), None)

    def _from_smiles(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute SMILESVec embeddings from SMILES using a pre-trained k-mer Word2Vec model.

        Auto-downloads the model from the artifacts bucket on first use.

        :param source: Drug feature source.
        :param entity_ids: Drug identifiers.
        :returns: Embedding matrix of shape (n_drugs, dim).
        :raises ValueError: If the source has no SMILES.
        :raises ImportError: If gensim is not installed.
        """
        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = f"Cannot obtain {self.storage_key}: no SMILES available."
            raise ValueError(msg)

        try:
            from gensim.models import KeyedVectors
        except ImportError as err:
            msg = "gensim is required for SMILESVec computation: pip install gensim"
            raise ImportError(msg) from err

        model_path = get_artifact(_SMILESVEC_MODEL_FILE)
        kv = KeyedVectors.load_word2vec_format(str(model_path), binary=False)
        k = self._k
        dim = kv.vector_size

        results = np.zeros((len(entity_ids), dim), dtype=np.float32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles.get(drug_id)
            if smi and isinstance(smi, str):
                results[i] = _smilesvec_embed(smi, kv, k=k, dim=dim)
            else:
                results[i] = np.nan
        return results


def _smilesvec_embed(smiles: str, kv: Any, *, k: int = 8, dim: int = 100) -> np.ndarray:
    """Compute a single SMILESVec embedding using k-mer averaging.

    :param smiles: SMILES string.
    :param kv: Gensim KeyedVectors model.
    :param k: Length of substrings (chemical words).
    :param dim: Embedding dimensionality.
    :returns: Embedding vector.
    """
    if len(smiles) < k:
        words = [smiles]
    else:
        words = [smiles[i : i + k] for i in range(len(smiles) - k + 1)]
    vecs = [kv[w] for w in words if w in kv.key_to_index]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)
