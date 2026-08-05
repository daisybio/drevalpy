"""Precomputed molecular graph drug featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, graph_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset


@register_drug_featurizer(
    "drugGraph",
    description="Precomputed PyG molecular graphs stored under the drug_graph view.",
    contract=FeatureFormat.GRAPH,
)
class DrugGraphFeaturizer(DrugFeaturizer):
    """Expose precomputed drug graphs for graph predictors."""

    def __init__(self, *, view: str = "drug_graph") -> None:
        """Store the graph view name and initialize empty caches.

        :param view: Feature view name containing graph payloads.
        """
        self._view = view
        self._graphs: dict[str, object] = {}
        self._output_dim = 0

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load precomputed DrugGNN graph artifacts.

        :param data_path: Parent directory for dataset artifacts.
        :param dataset_name: Dataset folder name.
        :param kwargs: Unused loader keyword arguments.
        :returns: Feature dataset mapping drug ids to graph payloads.
        :raises FileNotFoundError: If the graph directory is missing.
        :raises ValueError: If no graph files are found.
        """
        _ = cls, kwargs
        directory = Path(data_path) / dataset_name / "drug_graphs"
        if not directory.exists():
            raise FileNotFoundError(f"Drug graph directory not found at {directory}")
        graphs = {path.stem: torch.load(path, weights_only=False) for path in directory.glob("*.pt")}  # noqa: S614
        if not graphs:
            raise ValueError(f"No drug graphs loaded from {directory}")
        return FeatureDataset({drug_id: {"drug_graph": graph} for drug_id, graph in graphs.items()})

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> DrugGraphFeaturizer:
        """Cache graph payloads and infer node feature width from the first graph.

        :param features: Drug feature dataset.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param context: Unused featurizer fit context.
        :returns: Fitted featurizer instance.
        :raises KeyError: If the configured view is missing for a drug.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        self._graphs = {}
        for drug_id in ids:
            views = features.features[str(drug_id)]
            if self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_id!r}"
                raise KeyError(msg)
            self._graphs[str(drug_id)] = views[self._view]
        if self._graphs:
            first = next(iter(self._graphs.values()))
            self._output_dim = int(getattr(first, "num_node_features", 0))
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        """Return one graph payload per drug id.

        :param features: Drug feature dataset.
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
            views = features.features.get(drug_key)
            if views is None or self._view not in views:
                msg = f"View {self._view!r} missing for drug {drug_key!r}"
                raise KeyError(msg)
            graphs.append(views[self._view])
        payloads = np.empty(len(graphs), dtype=object)
        payloads[:] = graphs
        return payloads

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``drug_graph`` graph block.

        :param features: Drug feature dataset.
        :param entity_ids: Drug identifiers to transform.
        :returns: Mapping with one graph block.
        """
        return {"drug_graph": graph_feature_block(self.transform(features, entity_ids))}

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
