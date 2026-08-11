"""SMILESVec drug featurizer with auto-download of pre-trained Word2Vec model."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block

_logger = get_logger(__name__)

_SMILESVEC_MODEL_FILE = "drug.pubchem.canon.l8.ws20.txt"


@register_drug_featurizer(
    "smilesvec",
    description="SMILESVec drug embeddings computed on the fly via a pre-trained Word2Vec model.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    """SMILESVec featurizer using Word2Vec k-mer embeddings from PubChem corpus."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("smilesvec", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "smilesvec"
    input_views: ClassVar[tuple[str, ...]] = ("smilesvec",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str = "smilesvec", k: int = 8) -> None:
        """Initialize instance state.

        :param view: view.
        :param k: Length of SMILES subsequences (chemical words).
        """
        super().__init__(view=view)
        self._k = int(k)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "k": {"type": "categorical", "choices": [4, 6, 8, 10, 12], "default": 8},
        }

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute SMILESVec embeddings from SMILES using a pre-trained k-mer Word2Vec model.

        Auto-downloads the model from the artifacts bucket on first use.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Embedding matrix of shape (n_drugs, dim).
        """
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = f"Cannot obtain {self.storage_key}: no SMILES available."
            raise ValueError(msg)

        try:
            from gensim.models import KeyedVectors
        except ImportError as err:
            msg = "gensim is required for SMILESVec computation: pip install gensim"
            raise ImportError(msg) from err

        from drevalpy.data.artifacts import get_artifact

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

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Mapping with one numeric block.
        """
        return {
            "smilesvec": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }


def _smilesvec_embed(smiles: str, kv, *, k: int = 8, dim: int = 100) -> np.ndarray:
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
