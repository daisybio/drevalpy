"""ChemBERTa drug featurizer with on-the-fly computation fallback."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger

_logger = get_logger(__name__)

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
_CHEMBERTA_REVISION = "761d6a1"


@register_drug_featurizer(
    "chemberta",
    description="ChemBERTa embeddings loaded from pre-computed view or computed on the fly via transformers.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ChemBertaFeaturizer(ViewDrugFeaturizer):
    """ChemBERTa drug featurizer with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("chemberta", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "chemberta"
    input_views: ClassVar[tuple[str, ...]] = ("chemberta",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str = "chemberta", pooling: str = "mean", max_length: int = 512) -> None:
        """Initialize instance state.

        :param view: view.
        :param pooling: Token aggregation strategy ("mean", "cls", "max").
        :param max_length: Tokenizer truncation length.
        """
        super().__init__(view=view)
        self._pooling = pooling
        self._max_length = int(max_length)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "pooling": {"type": "categorical", "choices": ["mean", "cls", "max"], "default": "mean"},
            "max_length": {"type": "categorical", "choices": [64, 128, 256, 512], "default": 512},
        }

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute ChemBERTa embeddings from SMILES via pooled hidden states.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Embedding matrix of shape (n_drugs, hidden_dim).
        """
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = f"Cannot obtain {self.storage_key}: no SMILES available."
            raise ValueError(msg)

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as err:
            msg = (
                "transformers and torch are required for on-the-fly ChemBERTa computation: "
                "pip install transformers torch"
            )
            raise ImportError(msg) from err

        tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL, revision=_CHEMBERTA_REVISION)
        model = AutoModel.from_pretrained(_CHEMBERTA_MODEL, revision=_CHEMBERTA_REVISION)
        model.eval()

        embeddings = []
        for drug_id in entity_ids:
            smi = smiles.get(drug_id)
            if smi and isinstance(smi, str):
                inputs = tokenizer(smi, return_tensors="pt", truncation=True, max_length=self._max_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                embedding = self._pool(hidden_states)
            else:
                embedding = np.full(model.config.hidden_size, np.nan, dtype=np.float32)
            embeddings.append(embedding)

        return np.vstack(embeddings).astype(np.float32)

    def _pool(self, hidden_states) -> np.ndarray:
        """Apply pooling strategy to hidden states.

        :param hidden_states: Tensor of shape (1, seq_len, hidden_dim).
        :returns: Pooled embedding vector.
        """
        if self._pooling == "cls":
            return hidden_states[:, 0, :].squeeze(0).numpy()
        if self._pooling == "max":
            return hidden_states.max(dim=1).values.squeeze(0).numpy()
        return hidden_states.mean(dim=1).squeeze(0).numpy()

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Mapping with one numeric block.
        """
        return {
            "chemberta": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }
