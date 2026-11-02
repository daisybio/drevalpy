"""SMILESVec drug featurizer with auto-download of pre-trained Word2Vec model."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.featurizers._dense import PrecomputableFeaturizer
from drevalpy.data.artifacts import get_artifact
from drevalpy.registry.featurizer import register
from drevalpy.types.data.feature_source import FeatureSource, Side

_SMILESVEC_MODEL_FILE = "drug.pubchem.canon.l8.ws20.txt"


@register(
    "smilesvec",
    side=Side.DRUG,
    description="SMILESVec drug embeddings computed on the fly via a pre-trained Word2Vec model.",
    artifact_license="MIT",
)
class SmilesVecDrugFeaturizer(PrecomputableFeaturizer):
    """SMILESVec featurizer using Word2Vec k-mer embeddings from PubChem corpus."""

    block: ClassVar[str | None] = "smilesvec"
    input_views: ClassVar[tuple[str, ...]] = ("smilesvec",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "smilesvec", k: int = 8) -> None:
        """Configure the k-mer length of the on-the-fly embedding.

        :param view: Name of the view holding pre-computed embeddings.
        :param k: Length of SMILES subsequences (chemical words).
        """
        self._k = int(k)
        super().__init__(view, {"k": self._k})

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "k": {"type": "categorical", "choices": [4, 6, 8, 10, 12], "default": 8},
        }

    def compute(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Compute SMILESVec embeddings from SMILES using a pre-trained k-mer Word2Vec model.

        Auto-downloads the model from the artifacts bucket on first use.

        :param source: Drug feature source.
        :param ids: Drug identifiers.
        :returns: Embedding matrix of shape (n_drugs, dim).
        :raises ValueError: If the source has no SMILES.
        :raises ImportError: If gensim is not installed.
        """
        smiles = source.smiles(ids)
        if smiles is None:
            msg = f"Cannot compute {self._view}: no SMILES available."
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

        results = np.zeros((len(ids), dim), dtype=np.float32)
        for i, drug_id in enumerate(ids):
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
