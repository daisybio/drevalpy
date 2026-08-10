"""BPE PharmaFormer drug featurizer with proper fit/transform separation."""

from __future__ import annotations

import codecs
import tempfile
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger
from drevalpy.types.enums.literature_reference import LiteratureReference

_logger = get_logger(__name__)

_BPE_PHARMAFORMER_REFERENCE = LiteratureReference(
    repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    deviations=(
        "Set-dependent featurizer: BPE codes are learned from training SMILES "
        "at fit time and applied to any SMILES at transform time."
    ),
)


@register_drug_featurizer(
    "bpePharmaformer",
    description="BPE PharmaFormer token rows computed via fit/transform (set-dependent).",
    reference=_BPE_PHARMAFORMER_REFERENCE,
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BpePharmaformerDrugFeaturizer(DrugFeaturizer):
    """BPE PharmaFormer drug featurizer with proper fit/transform.

    Set-dependent: BPE codes are learned from the training SMILES during fit
    and applied to encode any SMILES during transform.
    """

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "bpe_smiles"
    input_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = False

    def __init__(self, *, view: str = "bpe_smiles", num_symbols: int = 10000, max_length: int = 128) -> None:
        """Initialize instance state.

        :param view: view.
        :param num_symbols: Number of BPE merge operations (vocabulary size).
        :param max_length: Maximum encoded sequence length.
        """
        self._view = view
        self._num_symbols = int(num_symbols)
        self._max_length = int(max_length)
        self._output_dim = self._max_length
        self._bpe = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "num_symbols": {"type": "categorical", "choices": [5000, 10000, 20000], "default": 10000},
            "max_length": {"type": "pow2", "low": 6, "high": 8, "default": 128},
        }

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> BpePharmaformerDrugFeaturizer:
        """Learn BPE codes from training SMILES.

        :param source: Feature source providing drug views.
        :param entity_ids: Training drug identifiers.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, ids)
        if smiles is None:
            msg = "Cannot learn BPE codes: no SMILES available."
            raise ValueError(msg)

        self._bpe = self._learn_bpe(smiles)
        self._output_dim = self._max_length
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Apply learned BPE codes to encode SMILES.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers to transform.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        """
        if self._bpe is None:
            msg = "BpePharmaformerDrugFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = "Cannot encode BPE: no SMILES available."
            raise ValueError(msg)
        return self._apply_bpe(smiles, entity_ids).astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Mapping with one numeric block.
        """
        return {
            "bpe_smiles": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=None,
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension.

        :returns: Always 128 (max_length).
        """
        return self._output_dim

    def _learn_bpe(self, smiles_series) -> object:
        """Learn BPE codes from a set of SMILES strings.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :returns: Fitted BPE object.
        """
        try:
            from subword_nmt.apply_bpe import BPE
            from subword_nmt.learn_bpe import learn_bpe
        except ImportError as err:
            msg = "subword-nmt is required for BPE computation: pip install subword-nmt"
            raise ImportError(msg) from err

        all_smiles = smiles_series.dropna()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp_file:
            tmp_path = tmp_file.name
            for smi in all_smiles:
                tmp_file.write(f"{smi}\n")

        bpe_codes_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".codes")
        bpe_codes_path = bpe_codes_file.name
        bpe_codes_file.close()

        import os
        from unittest.mock import patch

        try:
            with codecs.open(tmp_path, encoding="utf-8") as f_in:
                with codecs.open(bpe_codes_path, "w", encoding="utf-8") as f_out:
                    with patch("subword_nmt.learn_bpe.tqdm", side_effect=lambda it, *a, **kw: it):
                        learn_bpe(f_in, f_out, num_symbols=self._num_symbols, verbose=False)
        finally:
            os.unlink(tmp_path)

        with codecs.open(bpe_codes_path, encoding="utf-8") as f_in:
            bpe = BPE(f_in)

        os.unlink(bpe_codes_path)
        return bpe

    def _apply_bpe(self, smiles_series, entity_ids: np.ndarray) -> np.ndarray:
        """Apply stored BPE codes to encode SMILES strings.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param entity_ids: Drug identifiers.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        """
        results = np.zeros((len(entity_ids), self._max_length), dtype=np.int32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles_series.get(drug_id)
            if smi and isinstance(smi, str):
                bpe_processed = self._bpe.process_line(smi)
                encoded = [ord(char) for char in bpe_processed]
                if len(encoded) > self._max_length:
                    encoded = encoded[: self._max_length]
                else:
                    encoded = list(np.pad(encoded, (0, self._max_length - len(encoded)), "constant"))
                results[i] = encoded
        return results.astype(np.float32)
