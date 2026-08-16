"""Landmark gene featurizers for literature models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.components.featurizers.cell_line.gene_lists import gene_names_from_list_csv, resolve_gene_list_path
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


@register(
    "landmarkGenes",
    description="L1000 landmark genes with arcsinh and optional scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class LandmarkGenesFeaturizer(DenseViewCellLineFeaturizer):
    """Landmark genes featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True

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
        super().__init__(view=view)
        self._gene_list_stem = gene_list_stem
        self._standardize = standardize
        self._minmax_scale = minmax_scale
        self._arcsinh = arcsinh
        self._gene_indices: list[int] = []
        self._scaler: StandardScaler | None = None
        self._minmax: MinMaxScaler | None = None

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Select the landmark genes and fit the optional scalers.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to fit on.
        :returns: Number of selected genes.
        """
        self._gene_indices = _load_gene_indices(source, self._view, self._gene_list_stem)
        matrix = self._select_and_transform(self._raw_matrix(source, entity_ids))
        self._fit_scalers(matrix)
        return len(self._gene_indices)

    def _fit_scalers(self, matrix: np.ndarray) -> None:
        """Fit the standard and optional min-max scalers on *matrix*.

        :param matrix: Gene-subset matrix for the fit entities.
        """
        if not self._standardize:
            self._scaler = None
            self._minmax = None
            return
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        self._scaler = StandardScaler()
        self._scaler.fit(matrix)
        if self._minmax_scale:
            self._minmax = MinMaxScaler()
            self._minmax.fit(self._scaler.transform(matrix))
        else:
            self._minmax = None

    def _select_and_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Subset *matrix* to the selected genes and apply the optional arcsinh.

        :param matrix: Raw view matrix.
        :returns: Gene-subset matrix, before scaling.
        """
        selected = np.nan_to_num(matrix.astype(np.float64)[:, self._gene_indices], nan=0.0, posinf=0.0, neginf=0.0)
        return np.arcsinh(selected) if self._arcsinh else selected

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Subset, transform and scale *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Landmark-gene feature matrix.
        """
        _ = source
        result = self._select_and_transform(matrix)
        if self._scaler is not None:
            result = self._scaler.transform(result)
        if self._minmax is not None:
            result = self._minmax.transform(result)
        return result

    def _block_name(self) -> str:
        """Publish under the canonical gene-expression block name.

        :returns: Block name.
        """
        return "gene_expression"

    def _block_feature_names(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Return only the names of the selected landmark genes.

        :param source: Feature source providing view matrices.
        :returns: Selected gene names, or ``None`` when the source has none.
        """
        names = source.get_feature_names(self._view)
        if names is None:
            return None
        return tuple(str(names[index]) for index in self._gene_indices)

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
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        gene_indices = state.get("gene_indices")
        if isinstance(gene_indices, list):
            self._gene_indices = [int(index) for index in gene_indices]
        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler) or scaler is None:
            self._scaler = scaler
        minmax = state.get("minmax")
        if isinstance(minmax, MinMaxScaler) or minmax is None:
            self._minmax = minmax
        self._restore_dense_state(state)
        if not isinstance(state.get("output_dim"), int) and self._gene_indices:
            self._output_dim = len(self._gene_indices)

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        self._restore_landmark_identity(state)
        self._restore_landmark_fit_state(state)


@register(
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
