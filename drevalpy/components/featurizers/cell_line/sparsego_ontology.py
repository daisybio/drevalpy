"""SparseGO ontology-aligned cell-line featurizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.data_loading.multiomics import load_and_select_gene_features
from drevalpy.components.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line._sparsego_metadata import (
    SparseGOOntologyMetadata,
    attach_sparsego_ontology_metadata,
    read_sparsego_ontology_metadata,
)
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.literature.sparsego.utils import (
    load_mapping,
    load_ontology,
    pairs_in_layers,
    sort_pairs,
)
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.dataset import FeatureDataset


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

        :param config: Featurizer config whose hyperparameters select expression vs mutations.
        :returns: Single metadata-bearing numeric block for the active omics view.
        """
        hyperparams = getattr(config, "hyperparameters", None) or {}
        input_type = str(hyperparams.get("input_type", "expression") if isinstance(hyperparams, dict) else "expression")
        name = "mutations" if input_type == "mutations" else "gene_expression"
        return (BlockSpec(name, FeatureFormat.NUMERIC_MATRIX, metadata=True),)

    def __init__(self, *, input_type: str = "expression") -> None:
        """Validate *input_type* and initialize ontology metadata placeholders.

        :param input_type: Either ``expression`` or ``mutations``.
        :raises ValueError: If *input_type* is not supported.
        """
        if input_type not in {"expression", "mutations"}:
            raise ValueError("input_type must be 'expression' or 'mutations'")
        self._input_type = input_type
        self._view = "gene_expression" if input_type == "expression" else "mutations"
        self._layer_connections: list[np.ndarray] | None = None
        self._gene2id_mapping_ont: dict[str, int] | None = None
        self._ontology_gene_order: tuple[str, ...] = ()
        self._gene_dim_input = 0

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load, align, and annotate the active SparseGO omics feature view.

        :param data_path: Parent directory for dataset artifacts.
        :param dataset_name: Dataset folder name.
        :param kwargs: Loader options; ``input_type`` selects expression vs mutations.
        :returns: Feature dataset with ontology metadata attached.
        :raises FileNotFoundError: If SparseGO ontology files are missing.
        :raises ValueError: If ontology genes are absent from the active view.
        """
        input_type = str(kwargs.get("input_type", "expression"))
        view = "gene_expression" if input_type == "expression" else "mutations"
        root = Path(data_path) / dataset_name
        ontology_file, gene_index_file = root / "sparseGO_ont.txt", root / "gene2ind.txt"
        if not ontology_file.exists() or not gene_index_file.exists():
            raise FileNotFoundError(f"SparseGO requires {ontology_file.name} and {gene_index_file.name} in {root}")
        mapping = load_mapping(str(gene_index_file))
        order = tuple(sorted(mapping, key=mapping.__getitem__))
        features = load_and_select_gene_features(view, None, data_path, dataset_name)
        columns = {str(gene): index for index, gene in enumerate(features.meta_info[view])}
        missing = [gene for gene in order if gene not in columns]
        if missing:
            raise ValueError(f"Genes from gene2ind.txt missing in {view}: {missing[:5]}")
        indices = [columns[gene] for gene in order]
        for identifier in features.identifiers:
            features.features[str(identifier)][view] = features.features[str(identifier)][view][indices]
        features.meta_info[view] = np.asarray(order)
        graph, term_pairs, gene_term_pairs = load_ontology(str(ontology_file), mapping)
        sorted_pairs, level_list, level_numbers = sort_pairs(gene_term_pairs, term_pairs, graph, mapping)
        metadata = SparseGOOntologyMetadata(
            layer_connections=pairs_in_layers(sorted_pairs, level_list, level_numbers),
            gene2id_mapping_ont=mapping,
            ontology_gene_order=order,
            gene_dim_input=len(mapping),
        )
        attach_sparsego_ontology_metadata(features, metadata)
        return features

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> SparseGOOntologyFeaturizer:
        """Copy ontology metadata produced by ``load_features`` into fitted state.

        :param features: Feature dataset with ``_sparsego_ontology`` metadata.
        :param entity_ids: Unused.
        :param context: Unused featurizer fit context.
        :returns: Fitted featurizer instance.
        :raises ValueError: If ontology metadata is missing on *features*.
        """
        _ = entity_ids, context
        metadata = read_sparsego_ontology_metadata(features)
        if metadata is None:
            raise ValueError("SparseGO ontology metadata is missing; load features through sparsegoOntology")
        self._layer_connections = list(metadata["layer_connections"])
        self._gene2id_mapping_ont = dict(metadata["gene2id_mapping_ont"])
        self._ontology_gene_order = tuple(metadata["ontology_gene_order"])
        self._gene_dim_input = int(metadata["gene_dim_input"])
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return ontology-aligned omics matrix rows.

        :param features: Cell-line omics feature dataset.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix aligned to ontology gene order.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._gene_dim_input == 0:
            raise RuntimeError("SparseGOOntologyFeaturizer must be fit before transform")
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    def transform_blocks(self, features: FeatureDataset, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return an omics block with SparseGO ontology metadata attached.

        :param features: Cell-line omics feature dataset.
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
                self.transform(features, entity_ids),
                feature_names=feature_names_for_view(features, self._view),
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
            if input_type not in {"expression", "mutations"}:
                raise ValueError("input_type must be 'expression' or 'mutations'")
            self._input_type = input_type
            self._view = "gene_expression" if input_type == "expression" else "mutations"
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
