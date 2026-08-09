"""Precomputed molecular graph drug featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, graph_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.core.fitting.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "drugGraph",
    description="Precomputed PyG molecular graphs stored under the drug_graph view.",
    contract=FeatureFormat.GRAPH,
)
class DrugGraphFeaturizer(DrugFeaturizer):
    """Expose precomputed drug graphs for graph predictors."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("drug_graph", FeatureFormat.GRAPH),)
    input_views: ClassVar[tuple[str, ...]] = ("drug_graph",)

    def __init__(self, *, view: str = "drug_graph") -> None:
        """Store the graph view name and initialize empty caches.

        :param view: Feature view name containing graph payloads.
        """
        self._view = view
        self._graphs: dict[str, object] = {}
        self._output_dim = 0

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> DrugGraphFeaturizer:
        """Cache graph payloads and infer node feature width from the first graph.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param context: Unused featurizer fit context.
        :returns: Fitted featurizer instance.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._graphs = {}
        for drug_id in ids:
            graph = source.get_entity_view(str(drug_id), self._view)
            if graph is None:
                msg = f"View {self._view!r} missing for drug {drug_id!r}"
                raise KeyError(msg)
            self._graphs[str(drug_id)] = graph
        if self._graphs:
            first = next(iter(self._graphs.values()))
            self._output_dim = int(getattr(first, "num_node_features", 0))
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return one graph payload per drug id.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Object array of graph payloads.
        :raises KeyError: If the view is missing for a requested drug.
        """
        graphs: list[object] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._graphs:
                graphs.append(self._graphs[drug_key])
                continue
            graph = source.get_entity_view(drug_key, self._view)
            if graph is None:
                msg = f"View {self._view!r} missing for drug {drug_key!r}"
                raise KeyError(msg)
            graphs.append(graph)
        payloads = np.empty(len(graphs), dtype=object)
        payloads[:] = graphs
        return payloads

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``drug_graph`` graph block.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Mapping with one graph block.
        """
        return {"drug_graph": graph_feature_block(self.transform(source, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return node feature width inferred during ``fit``.

        :returns: Node feature dimensionality.
        """
        return self._output_dim

    @property
    def graph_by_drug(self) -> dict[str, object]:
        """Return fitted graph payloads keyed by drug id.

        :returns: Cached graph object per drug id.
        """
        return self._graphs
