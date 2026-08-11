"""Landmark gene featurizers for literature models."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.featurizers.cell_line.gene_lists import gene_names_from_list_csv, resolve_gene_list_path
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource


def _load_gene_indices(
    source: FeatureSource,
    view: str,
    gene_list_stem: str,
) -> list[int]:
    names = source.get_feature_names(view)
    if names is None:
        msg = f"FeatureSource has no feature names for view {view!r}"
        raise ValueError(msg)
    gene_list_path = resolve_gene_list_path(gene_list_stem)
    selected_genes = gene_names_from_list_csv(gene_list_path)
    gene_to_idx = {str(gene): index for index, gene in enumerate(names)}
    indices = [gene_to_idx[gene] for gene in selected_genes if gene in gene_to_idx]
    if not indices:
        msg = f"No genes from {gene_list_stem!r} matched view {view!r}"
        raise ValueError(msg)
    return indices


def _subset_matrix(
    source: FeatureSource,
    entity_ids: np.ndarray,
    *,
    view: str,
    gene_indices: list[int],
    scaler: StandardScaler | None = None,
    minmax: MinMaxScaler | None = None,
    arcsinh: bool = True,
) -> np.ndarray:
    matrix = stack_view_matrix(source, view, entity_ids).astype(np.float64)
    matrix = matrix[:, gene_indices]
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if arcsinh:
        matrix = np.arcsinh(matrix)
    if scaler is not None:
        matrix = scaler.transform(matrix)
    if minmax is not None:
        matrix = minmax.transform(matrix)
    return matrix.astype(np.float32)


@register_cell_line_featurizer(
    "landmarkGenes",
    description="L1000 landmark genes with arcsinh and optional scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class LandmarkGenesFeaturizer(CellLineFeaturizer):
    """Landmark genes featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        gene_list_stem: str = "landmark_genes",
        standardize: bool = True,
        minmax_scale: bool = False,
        arcsinh: bool = True,
    ) -> None:
        """Initialize instance state.

        :param view: view.
        :param gene_list_stem: gene list stem.
        :param standardize: standardize.
        :param minmax_scale: minmax scale.
        :param arcsinh: arcsinh.
        """
        self._view = view
        self._gene_list_stem = gene_list_stem
        self._standardize = standardize
        self._minmax_scale = minmax_scale
        self._arcsinh = arcsinh
        self._gene_indices: list[int] = []
        self._scaler: StandardScaler | None = None
        self._minmax: MinMaxScaler | None = None
        self._output_dim = 0
        self._is_fitted = False

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> LandmarkGenesFeaturizer:
        """Fit on training data.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, ids) if mdata is not None else None
        if precomputed is not None:
            self._output_dim = int(precomputed.shape[1])
            self._is_fitted = True
            return self
        self._gene_indices = _load_gene_indices(
            source,
            self._view,
            self._gene_list_stem,
        )
        self._output_dim = len(self._gene_indices)
        matrix = stack_view_matrix(source, self._view, ids).astype(np.float64)[:, self._gene_indices]
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if self._arcsinh:
            matrix = np.arcsinh(matrix)
        if self._standardize:
            self._scaler = StandardScaler()
            self._scaler.fit(matrix)
            if self._minmax_scale:
                z = self._scaler.transform(matrix)
                self._minmax = MinMaxScaler()
                self._minmax.fit(z)
            else:
                self._minmax = None
        else:
            self._scaler = None
            self._minmax = None
        self._is_fitted = True
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if not self._is_fitted:
            msg = "LandmarkGenesFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        return _subset_matrix(
            source,
            entity_ids,
            view=self._view,
            gene_indices=self._gene_indices,
            scaler=self._scaler,
            minmax=self._minmax,
            arcsinh=self._arcsinh,
        )

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if not self._is_fitted:
            msg = "LandmarkGenesFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        selected_names = None
        names = source.get_feature_names(self._view)
        if names is not None:
            selected_names = tuple(str(names[index]) for index in self._gene_indices)
        return {
            "gene_expression": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=selected_names,
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "standardize": {"type": "categorical", "choices": [True, False], "default": True},
            "minmax_scale": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if not self._is_fitted:
            return {}
        return {
            "view": self._view,
            "gene_list_stem": self._gene_list_stem,
            "standardize": self._standardize,
            "minmax_scale": self._minmax_scale,
            "arcsinh": self._arcsinh,
            "gene_indices": list(self._gene_indices),
            "scaler": self._scaler,
            "minmax": self._minmax,
            "output_dim": self._output_dim,
            "fitted": True,
        }

    def _restore_landmark_identity(self, state: dict[str, object]) -> None:
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        stem = state.get("gene_list_stem")
        if isinstance(stem, str):
            self._gene_list_stem = stem
        if "standardize" in state:
            self._standardize = bool(state["standardize"])
        if "minmax_scale" in state:
            self._minmax_scale = bool(state["minmax_scale"])
        if "arcsinh" in state:
            self._arcsinh = bool(state["arcsinh"])

    def _restore_landmark_fit_state(self, state: dict[str, object]) -> None:
        gene_indices = state.get("gene_indices")
        if isinstance(gene_indices, list):
            self._gene_indices = [int(index) for index in gene_indices]
        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler) or scaler is None:
            self._scaler = scaler
        minmax = state.get("minmax")
        if isinstance(minmax, MinMaxScaler) or minmax is None:
            self._minmax = minmax
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        elif self._gene_indices:
            self._output_dim = len(self._gene_indices)
        if state.get("fitted"):
            self._is_fitted = True

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        self._restore_landmark_identity(state)
        self._restore_landmark_fit_state(state)


@register_cell_line_featurizer(
    "landmarkGenesReduced",
    description="Reduced landmark gene set used by DrugGNN and PharmaFormer.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class LandmarkGenesReducedFeaturizer(LandmarkGenesFeaturizer):
    """Landmark genes reduced featurizer component."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        standardize: bool = False,
        minmax_scale: bool = False,
        arcsinh: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the reduced landmark featurizer variant.

        :param view: Omics view name (defaults to ``gene_expression``).
        :param standardize: Whether to z-score features after loading.
        :param minmax_scale: Whether to min-max scale features after loading.
        :param arcsinh: Whether to apply ``arcsinh`` transform after loading.
        :param kwargs: Ignored keyword arguments.
        """
        super().__init__(
            view=view,
            gene_list_stem="landmark_genes_reduced",
            standardize=standardize,
            minmax_scale=minmax_scale,
            arcsinh=arcsinh,
        )
