"""SparseGO ontology-aligned cell-line featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line._sparsego_metadata import (
    read_sparsego_ontology_metadata,
)
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_INPUT_TYPES = frozenset({"expression", "mutations"})


def _view_for_input_type(input_type: str) -> str:
    """Map a SparseGO ``input_type`` to the omics view it reads.

    :param input_type: Either ``expression`` or ``mutations``.
    :returns: Omics view name backing that input type.
    """
    return "mutations" if input_type == "mutations" else "gene_expression"


@register_cell_line_featurizer(
    "sparsegoOntology",
    description="SparseGO ontology-aligned expression or mutation inputs.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SparseGOOntologyFeaturizer(CellLineFeaturizer):
    """Align an active omics view with the SparseGO ontology gene ordering."""

    output_block_specs = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX, metadata=True),)

    @classmethod
    def output_block_specs_for_config(cls, config: Any) -> tuple[BlockSpec, ...]:
        """Name the active SparseGO block from ``input_type``.

        :param config: Featurizer template; uses space default when unresolved.
        :returns: Single metadata-bearing numeric block for the active omics view.
        """
        raw_space = getattr(config, "hyperparameter_space", None) or {}
        space = dict(raw_space) if isinstance(raw_space, Mapping) else {}
        if not space:
            space = dict(cls.get_hyperparameter_space())
        spec = space.get("input_type")
        if isinstance(spec, Mapping) and "default" in spec:
            input_type = str(spec["default"])
        else:
            input_type = "expression"
        name = _view_for_input_type(input_type)
        return (BlockSpec(name, FeatureFormat.NUMERIC_MATRIX, metadata=True),)

    @classmethod
    def resolve_input_views(cls, **kwargs: Any) -> tuple[str, ...]:
        """Return the omics view selected by ``input_type``.

        :param kwargs: Featurizer kwargs; ``input_type`` selects expression vs mutations.
        :returns: Single-element tuple with the active omics view.
        """
        return (_view_for_input_type(str(kwargs.get("input_type", "expression"))),)

    def __init__(self, *, input_type: str = "expression") -> None:
        """Validate *input_type* and initialize ontology metadata placeholders.

        :param input_type: Either ``expression`` or ``mutations``.
        :raises ValueError: If *input_type* is not supported.
        """
        if input_type not in _INPUT_TYPES:
            raise ValueError("input_type must be 'expression' or 'mutations'")
        self._input_type = input_type
        self._view = _view_for_input_type(input_type)
        self._layer_connections: list[np.ndarray] | None = None
        self._gene2id_mapping_ont: dict[str, int] | None = None
        self._ontology_gene_order: tuple[str, ...] = ()
        self._gene_dim_input = 0

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> SparseGOOntologyFeaturizer:
        """Copy ontology metadata produced by ``load_features`` into fitted state.

        :param source: Feature source with ``sparsego_ontology`` metadata.
        :param entity_ids: Unused.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        :raises ValueError: If ontology metadata is missing on *source*.
        """
        _ = entity_ids, pair_expanded_ids, pair_expanded_es_ids
        metadata = read_sparsego_ontology_metadata(source)
        if metadata is None:
            raise ValueError("SparseGO ontology metadata is missing; load features through sparsegoOntology")
        self._layer_connections = list(metadata["layer_connections"])
        self._gene2id_mapping_ont = dict(metadata["gene2id_mapping_ont"])
        self._ontology_gene_order = tuple(metadata["ontology_gene_order"])
        self._gene_dim_input = int(metadata["gene_dim_input"])
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return ontology-aligned omics matrix rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix aligned to ontology gene order.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._gene_dim_input == 0:
            raise RuntimeError("SparseGOOntologyFeaturizer must be fit before transform")
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        return source.get_view_matrix(self._view, entity_ids).astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return an omics block with SparseGO ontology metadata attached.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping with one metadata-rich numeric block.
        """
        metadata: dict[str, Any] = {
            "layer_connections": self._layer_connections,
            "gene2id_mapping_ont": self._gene2id_mapping_ont,
            "ontology_gene_order": self._ontology_gene_order,
            "gene_dim_input": self._gene_dim_input,
        }
        return {
            self._view: numeric_feature_block(
                self.transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
                metadata=metadata,
            )
        }

    @property
    def output_dim(self) -> int:
        """Return ontology gene dimensionality.

        :returns: Number of ontology-aligned genes.
        """
        return self._gene_dim_input

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable SparseGO input type.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": "expression"}}

    def get_state(self) -> dict[str, object]:
        """Serialize ontology metadata and input type.

        :returns: Fitted state mapping, or empty dict before fitting.
        """
        if self._gene_dim_input == 0:
            return {}
        return {
            "input_type": self._input_type,
            "layer_connections": self._layer_connections,
            "gene2id_mapping_ont": self._gene2id_mapping_ont,
            "ontology_gene_order": self._ontology_gene_order,
            "gene_dim_input": self._gene_dim_input,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore ontology metadata from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        :raises ValueError: If stored ``input_type`` is invalid.
        """
        input_type = state.get("input_type")
        if isinstance(input_type, str):
            if input_type not in _INPUT_TYPES:
                raise ValueError("input_type must be 'expression' or 'mutations'")
            self._input_type = input_type
            self._view = _view_for_input_type(input_type)
        mapping = state.get("gene2id_mapping_ont")
        if isinstance(mapping, dict):
            self._gene2id_mapping_ont = {str(key): int(value) for key, value in mapping.items()}
        order = state.get("ontology_gene_order")
        if isinstance(order, tuple):
            self._ontology_gene_order = tuple(str(gene) for gene in order)
        connections = state.get("layer_connections")
        if isinstance(connections, list):
            self._layer_connections = [np.asarray(connection) for connection in connections]
        gene_dim_input = state.get("gene_dim_input")
        if isinstance(gene_dim_input, int):
            self._gene_dim_input = gene_dim_input
