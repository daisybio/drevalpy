"""Landmark gene featurizers for literature models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler


@register(
    "landmarkGenes",
    description="L1000 landmark genes with arcsinh and optional scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class LandmarkGenesFeaturizer(DenseViewSampleFeaturizer):
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
        """Configure the landmark-gene selection and preprocessing.

        :param view: Name of the gene-expression view to read.
        :param gene_list_stem: Name of the shipped gene list to keep; the view must hold every gene of it.
        :param standardize: Z-score the selected genes with a scaler fit on the training samples.
        :param minmax_scale: Min-max scale after standardization; only applied when *standardize* is set.
        :param arcsinh: Apply ``arcsinh`` to the selected genes before scaling.
        """
        super().__init__(view=view)
        self._gene_list_stem = gene_list_stem
        self._standardize = standardize
        self._minmax_scale = minmax_scale
        self._arcsinh = arcsinh
        self._scaler: StandardScaler | None = None
        self._minmax: MinMaxScaler | None = None

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...]:
        """Bind the transform to the genes of the list, in list order.

        :param source: Feature source providing view matrices.
        :returns: Gene names in gene-list order.
        """
        return select_genes(source, self._view, self._gene_list_stem)

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the optional scalers on the landmark-gene columns.

        :param source: Feature source providing view matrices.
        :param entity_ids: Sample identifiers to fit on.
        :returns: Number of selected genes.
        """
        self._fit_scalers(self._transform_genes(self._raw_matrix(source, entity_ids)))
        return len(self._view_names or ())

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

    def _transform_genes(self, matrix: np.ndarray) -> np.ndarray:
        """Clean the landmark-gene columns of *matrix* and apply the optional arcsinh.

        :param matrix: Landmark-gene matrix in fitted column order.
        :returns: Cleaned matrix, before scaling.
        """
        selected = np.nan_to_num(matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return np.arcsinh(selected) if self._arcsinh else selected

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Transform and scale the landmark-gene columns of *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Landmark-gene matrix for the requested entity IDs.
        :returns: Landmark-gene feature matrix.
        """
        _ = source
        result = self._transform_genes(matrix)
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
        :returns: Selected gene names, or ``None`` before fit.
        """
        _ = source
        return self._view_names

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the searchable hyperparameters."""
        return {
            "standardize": {"type": "categorical", "choices": [True, False], "default": True},
            "minmax_scale": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state."""
        if not self._is_fitted:
            return {}
        return {
            **self._dense_state(),
            "gene_list_stem": self._gene_list_stem,
            "standardize": self._standardize,
            "minmax_scale": self._minmax_scale,
            "arcsinh": self._arcsinh,
            "scaler": self._scaler,
            "minmax": self._minmax,
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

        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler) or scaler is None:
            self._scaler = scaler
        minmax = state.get("minmax")
        if isinstance(minmax, MinMaxScaler) or minmax is None:
            self._minmax = minmax
        self._restore_dense_state(state)
        if not isinstance(state.get("output_dim"), int) and self._view_names:
            self._output_dim = len(self._view_names)

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
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
        r"""Initialize the reduced landmark featurizer variant.

        :param view: Omics view name (defaults to ``gene_expression``).
        :param standardize: Whether to z-score features after loading.
        :param minmax_scale: Whether to min-max scale features after loading.
        :param arcsinh: Whether to apply ``arcsinh`` transform after loading.
        :param \*\*kwargs: Ignored.
        """
        super().__init__(
            view=view,
            gene_list_stem="landmark_genes_reduced",
            standardize=standardize,
            minmax_scale=minmax_scale,
            arcsinh=arcsinh,
        )
