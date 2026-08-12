"""ChemBERTa drug featurizer with on-the-fly computation fallback."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.log import get_logger
from drevalpy.registry.drug_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_logger = get_logger(__name__)

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
_CHEMBERTA_REVISION = "761d6a1"

# Weights are mirrored to the drevalpy artifacts bucket rather than pulled from the
# HuggingFace Hub: Hub downloads are rate-limited per source IP, which fails en masse
# when hundreds of pipeline workers behind one NAT gateway start with a cold cache.
_CHEMBERTA_ARTIFACT = "chemberta_zinc_base_v1_761d6a1"
_CHEMBERTA_ARTIFACT_FILES = (
    "config.json",
    "merges.txt",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


@lru_cache(maxsize=1)
def load_chemberta() -> tuple[Any, Any]:
    """Return the cached ChemBERTa tokenizer and model.

    The weights are fetched once from the artifacts location and then reused for
    the lifetime of the process, so repeated HPO trials do not reload them.

    :returns: Tuple of (tokenizer, model) with the model in eval mode.
    :raises ImportError: If transformers or torch are unavailable.
    :raises RuntimeError: If the weights cannot be fetched or loaded.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as err:
        msg = "transformers and torch are required for on-the-fly ChemBERTa computation: pip install transformers torch"
        raise ImportError(msg) from err

    from drevalpy.data.artifacts import get_artifact_dir, get_artifacts_uri

    try:
        model_dir = str(get_artifact_dir(_CHEMBERTA_ARTIFACT, _CHEMBERTA_ARTIFACT_FILES))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    except (ImportError, RuntimeError):
        raise
    except Exception as err:
        msg = (
            f"Could not load ChemBERTa weights ({_CHEMBERTA_MODEL} @ {_CHEMBERTA_REVISION}) "
            f"from artifact {_CHEMBERTA_ARTIFACT!r} at {get_artifacts_uri()!r}: {err}. "
            "Check credentials and connectivity for that location, or point "
            "DREVALPY_ARTIFACTS_URI at a reachable mirror."
        )
        raise RuntimeError(msg) from err

    model.eval()
    return tokenizer, model


@register(
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
        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = f"Cannot obtain {self.storage_key}: no SMILES available."
            raise ValueError(msg)

        import torch

        tokenizer, model = load_chemberta()

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
