"""Morgan fingerprint drug featurizer."""

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
    "fingerprints",
    description="Morgan fingerprints loaded from pre-computed view or computed on the fly via rdkit.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class FingerprintsFeaturizer(ViewDrugFeaturizer):
    """Morgan fingerprints featurizer with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("fingerprints", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "morgan_fingerprint"
    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)

    def __init__(self, *, view: str = "morgan_fingerprint") -> None:
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
    ) -> FingerprintsFeaturizer:
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

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into fingerprint features.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Feature matrix.
        """
        return self._get_or_compute(source, entity_ids).astype(np.float32)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Mapping with one numeric block.
        """
        return {
            "fingerprints": numeric_feature_block(
                self.transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }

    def _get_or_compute(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Try pre-computed fetch, fall back to on-the-fly computation.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Fingerprint matrix.
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
            return self._compute_from_smiles(smiles, entity_ids)
        msg = f"Cannot obtain {self.storage_key}: no pre-computed data, view, or SMILES available."
        raise ValueError(msg)

    def _compute_from_smiles(self, smiles_series, entity_ids: np.ndarray) -> np.ndarray:
        """Compute Morgan fingerprints from SMILES.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param entity_ids: Drug identifiers.
        :returns: Fingerprint bit matrix of shape (n_drugs, 2048).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as err:
            msg = "rdkit is required for on-the-fly fingerprint computation: pip install rdkit"
            raise ImportError(msg) from err

        n_bits = 2048
        results = np.zeros((len(entity_ids), n_bits), dtype=np.float32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles_series.get(drug_id)
            if smi and isinstance(smi, str):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                    results[i] = np.array(fp, dtype=np.float32)
                else:
                    results[i] = np.nan
            else:
                results[i] = np.nan
        return results
