"""SparseGO ontology-aligned sample featurizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.featurizers.sample._sparsego_metadata import (
    SparseGOOntologyMetadata,
    read_sparsego_ontology_metadata,
)
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

_INPUT_TYPES = frozenset({"expression", "mutations"})


def _view_for_input_type(input_type: str) -> str:
    """Map a SparseGO ``input_type`` to the omics view it reads.

    :param input_type: Either ``expression`` or ``mutations``.
    :returns: Omics view name backing that input type.
    """
    return "mutations" if input_type == "mutations" else "gene_expression"


@register("sparsegoOntology", description="SparseGO ontology-aligned expression or mutation inputs.")
class SparseGOOntologyFeaturizer(Featurizer):
    """Align an active omics view with the SparseGO ontology gene ordering."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})

    @classmethod
    def _config_view(cls, config: Mapping[str, Any]) -> str:
        """Return the view selected by the ``input_type`` of *config*, else by the space default.

        :param config: Construction kwargs of the featurizer.
        :returns: ``mutations`` or ``gene_expression``.
        """
        input_type = config.get("input_type", cls.get_hyperparameter_space()["input_type"]["default"])
        return _view_for_input_type(str(input_type))

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Name the active SparseGO block from ``input_type``.

        :param config: Construction kwargs of the featurizer.
        :returns: Single metadata-bearing numeric block for the active omics view.
        """
        return (BlockSpec(cls._config_view(config), FeatureFormat.NUMERIC_MATRIX),)

    @classmethod
    def views(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the omics view selected by ``input_type``.

        :param config: Construction kwargs of the featurizer.
        :returns: Single-element tuple with the active omics view.
        """
        return (cls._config_view(config),)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable SparseGO input type.

        :returns: Hyperparameter search space mapping.
        """
        return {"input_type": {"type": "categorical", "choices": ["expression", "mutations"], "default": "expression"}}

    def __init__(self, *, input_type: str = "expression") -> None:
        """Validate *input_type*.

        :param input_type: Either ``expression`` or ``mutations``.
        :raises ValueError: If *input_type* is not supported.
        """
        if input_type not in _INPUT_TYPES:
            raise ValueError("input_type must be 'expression' or 'mutations'")
        self._input_type = input_type
        self._view = _view_for_input_type(input_type)
        self._metadata: SparseGOOntologyMetadata | None = None

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the active view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Copy the ontology metadata produced by ``load_features`` into fitted state.

        :param source: Feature source with ``sparsego_ontology`` metadata.
        :param train_ids: Unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = train_ids, train_pair_ids, es_pair_ids
        self._metadata = _required_metadata(source)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the active view's rows for *ids* in ontology gene order, with the ontology attached.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: Result with one metadata-rich numeric block.
        :raises RuntimeError: If called before ``fit``.
        :raises ValueError: If the view has no feature names or lacks ontology genes.
        """
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("SparseGOOntologyFeaturizer must be fit before transform")
        names = source.feature_names(self._view)
        if names is None:
            raise ValueError("SparseGO requires named input features to align ontology genes")
        columns = {name: index for index, name in enumerate(names)}
        missing = set(metadata["ontology_gene_order"]) - columns.keys()
        if missing:
            raise ValueError(f"SparseGO input is missing ontology genes: {sorted(missing)}")
        indices = [columns[name] for name in metadata["ontology_gene_order"]]
        return self._result(ids, source.view(self._view, ids)[:, indices], metadata)

    def _result(self, ids: np.ndarray, matrix: np.ndarray, metadata: SparseGOOntologyMetadata) -> FeaturizerResult:
        """Wrap *matrix* as the active view's block, annotated with *metadata*.

        :param ids: Sample IDs, one per row of *matrix*.
        :param matrix: Ontology-aligned matrix.
        :param metadata: Ontology metadata to attach.
        :returns: Result with one metadata-rich numeric block.
        """
        block = numeric_feature_block(
            matrix.astype(np.float32),
            feature_names=metadata["ontology_gene_order"],
            annotations=dict(metadata),
        )
        return FeaturizerResult(ids, {self._view: block}, (self._view,))


def _required_metadata(source: FeatureSource) -> SparseGOOntologyMetadata:
    """Return the ontology metadata of *source*.

    :param source: Feature source with ``sparsego_ontology`` metadata.
    :returns: The ontology metadata.
    :raises ValueError: If ontology metadata is missing on *source*.
    """
    metadata = read_sparsego_ontology_metadata(source)
    if metadata is None:
        raise ValueError("SparseGO ontology metadata is missing; load features through sparsegoOntology")
    return metadata
