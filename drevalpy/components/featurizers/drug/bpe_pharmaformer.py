"""BPE PharmaFormer drug featurizer with on-the-fly computation fallback."""

from __future__ import annotations

import codecs
import tempfile
from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger
from drevalpy.types.enums.literature_reference import LiteratureReference

_logger = get_logger(__name__)

_BPE_PHARMAFORMER_REFERENCE = LiteratureReference(
    repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    deviations=(
        "Consumes precomputed BPE token rows from the bpe_smiles view; "
        "offline embedding generation is implemented in "
        "drevalpy.data.featurizer.create_pharmaformer_drug_embeddings."
    ),
)


@register_drug_featurizer(
    "bpePharmaformer",
    description="BPE PharmaFormer token rows loaded from pre-computed view or computed on the fly.",
    reference=_BPE_PHARMAFORMER_REFERENCE,
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BpePharmaformerDrugFeaturizer(ViewDrugFeaturizer):
    """BPE PharmaFormer drug featurizer with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "bpe_smiles"
    input_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)

    def __init__(self, *, view: str = "bpe_smiles") -> None:
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
    ) -> BpePharmaformerDrugFeaturizer:
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
        """Transform inputs into BPE-encoded SMILES.

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
            "bpe_smiles": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }

    def _get_or_compute(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Try pre-computed fetch, fall back to on-the-fly computation.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: BPE token matrix.
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
        """Compute BPE-encoded SMILES on the fly.

        Learns BPE codes from all available SMILES in the dataset, then encodes the
        requested entities.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param entity_ids: Drug identifiers.
        :param source: Feature source for access to all SMILES.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        """
        try:
            from subword_nmt.apply_bpe import BPE
            from subword_nmt.learn_bpe import learn_bpe
        except ImportError as err:
            msg = "subword-nmt is required for on-the-fly BPE computation: pip install subword-nmt"
            raise ImportError(msg) from err

        num_symbols = 10000
        max_length = 128

        mdata = getattr(source, "mdata", None)
        if mdata is not None:
            all_smiles = mdata.mod["response"].var["canonical_smiles"].dropna()
        else:
            all_smiles = smiles_series.dropna()

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp_file:
            tmp_path = tmp_file.name
            for smi in all_smiles:
                tmp_file.write(f"{smi}\n")

        bpe_codes_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".codes")
        bpe_codes_path = bpe_codes_file.name
        bpe_codes_file.close()

        try:
            with codecs.open(tmp_path, encoding="utf-8") as f_in:
                with codecs.open(bpe_codes_path, "w", encoding="utf-8") as f_out:
                    learn_bpe(f_in, f_out, num_symbols=num_symbols)
        finally:
            import os

            os.unlink(tmp_path)

        with codecs.open(bpe_codes_path, encoding="utf-8") as f_in:
            bpe = BPE(f_in)

        import os

        os.unlink(bpe_codes_path)

        results = np.zeros((len(entity_ids), max_length), dtype=np.int32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles_series.get(drug_id)
            if smi and isinstance(smi, str):
                bpe_processed = bpe.process_line(smi)
                encoded = [ord(char) for char in bpe_processed]
                if len(encoded) > max_length:
                    encoded = encoded[:max_length]
                else:
                    encoded = list(np.pad(encoded, (0, max_length - len(encoded)), "constant"))
                results[i] = encoded

        return results.astype(np.float32)
