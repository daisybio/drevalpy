"""SMILESVec drug featurizer with auto-download of pre-trained Word2Vec model.

The model is a Word2Vec file in the plain-text format (a ``<vocabulary size> <dimension>`` header, then one
``<word> <value> ... <value>`` line per word); :func:`_read_word2vec_text` reads it, so no Word2Vec library is needed.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from upath import UPath

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
        """
        smiles = source.smiles(ids)
        if smiles is None:
            msg = f"Cannot compute {self._view}: no SMILES available."
            raise ValueError(msg)

        model_path = get_artifact(_SMILESVEC_MODEL_FILE)
        key_to_index, vectors = _read_word2vec_text(model_path)
        k = self._k
        dim = vectors.shape[1]

        results = np.zeros((len(ids), dim), dtype=np.float32)
        for i, smi in enumerate(smiles):
            if smi and isinstance(smi, str):
                results[i] = _smilesvec_embed(smi, key_to_index, vectors, k=k)
            else:
                results[i] = np.nan
        return results


def _read_word2vec_text(path: str | UPath) -> tuple[dict[str, int], np.ndarray]:
    """Read a Word2Vec model in the plain-text format.

    :param path: File with a ``<vocabulary size> <dimension>`` header line and one ``<word> <values>`` line per word.
    :returns: The row of each word in the matrix, and the matrix of shape (vocabulary size, dimension).
    """
    with UPath(path).open(encoding="utf-8") as handle:
        vocabulary_size, dim = (int(n) for n in handle.readline().split())
        key_to_index: dict[str, int] = {}
        vectors = np.empty((vocabulary_size, dim), dtype=np.float32)
        for line in handle:
            word, *values = line.rstrip("\n").split(" ")
            vectors[len(key_to_index)] = values[:dim]
            key_to_index[word] = len(key_to_index)
    return key_to_index, vectors


def _smilesvec_embed(smiles: str, key_to_index: dict[str, int], vectors: np.ndarray, *, k: int = 8) -> np.ndarray:
    """Compute a single SMILESVec embedding using k-mer averaging.

    :param smiles: SMILES string.
    :param key_to_index: Row of each k-mer in ``vectors``.
    :param vectors: Embedding of each k-mer, one row per word.
    :param k: Length of substrings (chemical words).
    :returns: Mean embedding of the k-mers in the model, zeros if none is.
    """
    if len(smiles) < k:
        words = [smiles]
    else:
        words = [smiles[i : i + k] for i in range(len(smiles) - k + 1)]
    rows = [key_to_index[w] for w in words if w in key_to_index]
    if not rows:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    return vectors[rows].mean(axis=0).astype(np.float32)
