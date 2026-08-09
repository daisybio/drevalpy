"""MolGNet drug featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, ragged_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.core.fitting.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "molgnet",
    description="Precomputed MolGNet drug embeddings for DIPK.",
    contract=FeatureFormat.RAGGED_SEQUENCE,
)
class MolGNetDrugFeaturizer(DrugFeaturizer):
    """Expose variable-size MolGNet tensors without stacking into one dense matrix."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE),
    )
    input_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)

    def __init__(self, *, view: str = "molgnet_features") -> None:
        """Store the MolGNet view name and initialize empty caches.

        :param view: Feature view name containing MolGNet tensors.
        """
        self._view = view
        self._features_by_drug: dict[str, np.ndarray] = {}
        self._output_dim = 0

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> MolGNetDrugFeaturizer:
        """Cache MolGNet tensors and infer embedding width.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param context: Unused featurizer fit context.
        :returns: Fitted featurizer instance.
        :raises KeyError: If the configured view is missing for a drug.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._features_by_drug = {}
        for drug_id in ids:
            entity_view = source.get_entity_view(str(drug_id), self._view)
            if entity_view is None:
                msg = f"View {self._view!r} missing for drug {drug_id!r}"
                raise KeyError(msg)
            self._features_by_drug[str(drug_id)] = np.asarray(entity_view)
        if self._features_by_drug:
            first = next(iter(self._features_by_drug.values()))
            self._output_dim = int(first.shape[1]) if first.ndim == 2 else int(first.size)
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return one MolGNet tensor per drug id.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Object array of MolGNet embedding tensors.
        :raises KeyError: If the view is missing for a requested drug.
        """
        rows: list[np.ndarray] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._features_by_drug:
                rows.append(self._features_by_drug[drug_key])
                continue
            entity_view = source.get_entity_view(drug_key, self._view)
            if entity_view is None:
                msg = f"View {self._view!r} missing for drug {drug_key!r}"
                raise KeyError(msg)
            rows.append(np.asarray(entity_view))
        return np.array(rows, dtype=object)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``molgnet_features`` ragged block.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Mapping with one ragged block.
        """
        return {"molgnet_features": ragged_feature_block(self.transform(source, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return embedding width inferred during ``fit``.

        :returns: MolGNet embedding dimensionality.
        """
        return self._output_dim
