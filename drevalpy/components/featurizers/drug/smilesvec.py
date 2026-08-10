"""SMILESVec drug featurizer with on-the-fly computation fallback."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger

_logger = get_logger(__name__)


@register_drug_featurizer(
    "smilesvec",
    description="SMILESVec drug embeddings loaded from pre-computed view or computed on the fly via gensim.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SmilesVecDrugFeaturizer(ViewDrugFeaturizer):
    """SMILESVec featurizer with on-the-fly fallback using Word2Vec k-mer embeddings."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("smilesvec", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "smilesvec"
    input_views: ClassVar[tuple[str, ...]] = ("smilesvec",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str = "smilesvec") -> None:
        """Initialize instance state.

        :param view: view.
        """
        super().__init__(view=view)

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> SmilesVecDrugFeaturizer:
        """Fit on training data, falling back to on-the-fly computation.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        matrix = self._get_or_compute(source, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into SMILESVec embeddings.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Feature matrix.
        """
        return self._get_or_compute(source, entity_ids).astype(np.float32)

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

    def _get_or_compute(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Try pre-computed fetch, fall back to on-the-fly computation.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Embedding matrix.
        """
        mdata = getattr(source, "mdata", None)
        if mdata is not None:
            precomputed = self.fetch(mdata, entity_ids)
            if precomputed is not None:
                return precomputed
        from drevalpy.components.featurizers._matrix import stack_view_matrix

        try:
            return stack_view_matrix(source, self._view, entity_ids)
        except (KeyError, TypeError, ValueError):
            pass
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is not None:
            _logger.warning("Computing %s on the fly. Consider ds.precompute().", self.storage_key)
            return self._compute_from_smiles(smiles, entity_ids, source)
        msg = f"Cannot obtain {self.storage_key}: no pre-computed data, view, or SMILES available."
        raise ValueError(msg)

    def _compute_from_smiles(self, smiles_series, entity_ids: np.ndarray, source: FeatureSource) -> np.ndarray:
        """Compute SMILESVec embeddings from SMILES using k-mer Word2Vec model.

        Requires a pre-trained Word2Vec model stored in mdata.uns["smilesvec_model_path"].

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param entity_ids: Drug identifiers.
        :param source: Feature source for metadata access.
        :returns: Embedding matrix of shape (n_drugs, dim).
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError as err:
            msg = "gensim is required for on-the-fly SMILESVec computation: pip install gensim"
            raise ImportError(msg) from err

        mdata = getattr(source, "mdata", None)
        model_path = None
        if mdata is not None and "smilesvec_model_path" in mdata.uns:
            model_path = mdata.uns["smilesvec_model_path"]

        if model_path is None:
            msg = (
                "SMILESVec requires a pre-trained Word2Vec model. "
                "Store the path in mdata.uns['smilesvec_model_path'] or pre-compute the embeddings."
            )
            raise ValueError(msg)

        kv = KeyedVectors.load_word2vec_format(str(model_path), binary=False)
        k = 8
        dim = kv.vector_size

        results = np.zeros((len(entity_ids), dim), dtype=np.float32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles_series.get(drug_id)
            if smi and isinstance(smi, str):
                results[i] = _smilesvec_embed(smi, kv, k=k, dim=dim)
            else:
                results[i] = np.nan
        return results


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
