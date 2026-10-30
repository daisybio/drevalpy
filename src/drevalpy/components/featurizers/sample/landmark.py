"""Landmark gene featurizers for literature models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

_BLOCK = "gene_expression"


@register("landmarkGenes", description="L1000 landmark genes with arcsinh and optional scaling.")
class LandmarkGenesFeaturizer(Featurizer):
    """Landmark genes featurizer component."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        gene_list_stem: str = "landmark_genes",
        standardize: bool = True,
        minmax_scale: bool = False,
        arcsinh: bool = True,
        fit_on_pairs: bool = False,
    ) -> None:
        """Configure the landmark-gene selection and preprocessing.

        :param view: Name of the gene-expression view to read.
        :param gene_list_stem: Name of the shipped gene list to keep; the view must hold every gene of it.
        :param standardize: Z-score the selected genes with a scaler fit on the training samples.
        :param minmax_scale: Min-max scale after standardization; only applied when *standardize* is set.
        :param arcsinh: Apply ``arcsinh`` to the selected genes before scaling.
        :param fit_on_pairs: Fit the scalers on the training pairs, so a sample counts once per pair, as the
            DrEvalPy 1.6.1 PaccMann did, instead of once per sample.
        """
        self._view = view
        self._gene_list_stem = gene_list_stem
        self._standardize = standardize
        self._minmax_scale = minmax_scale
        self._arcsinh = arcsinh
        self._fit_on_pairs = fit_on_pairs
        self._names: tuple[str, ...] | None = None
        self._scaler: StandardScaler | None = None
        self._minmax: MinMaxScaler | None = None

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Publish under the canonical gene-expression block name, whatever view is read.

        :param config: Unused.
        :returns: The ``gene_expression`` block.
        """
        _ = config
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the searchable hyperparameters."""
        return {
            "standardize": {"type": "categorical", "choices": [True, False], "default": True},
            "minmax_scale": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def stored_params(self) -> Mapping[str, Any] | None:
        """Match the parameter-free stored variant.

        :returns: An empty mapping.
        """
        return {}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap a stored landmark matrix as the ``gene_expression`` block.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, one per row of *matrix*.
        :param matrix: Stored landmark-gene matrix.
        :returns: One ``gene_expression`` block named by the genes of the list.
        """
        return dense_result(ids, _BLOCK, matrix, select_genes(source, self._view, self._gene_list_stem))

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Bind the transform to the genes of the list and fit the optional scalers.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs; the scalers' rows unless ``fit_on_pairs`` is set.
        :param train_pair_ids: Sample ID of every training pair; the scalers' rows when ``fit_on_pairs``
            is set and they are given.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = es_pair_ids
        self._names = select_genes(source, self._view, self._gene_list_stem)
        ids = train_pair_ids if self._fit_on_pairs and train_pair_ids is not None else train_ids
        self._fit_scalers(self._transform_genes(view_matrix(source, self._view, ids, self._names)))
        return self

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

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Transform and scale the landmark-gene columns of the rows of *ids*.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One ``gene_expression`` block named by the selected genes.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._names is None:
            msg = f"{type(self).__name__} must be fit before transform"
            raise RuntimeError(msg)
        result = self._transform_genes(view_matrix(source, self._view, ids, self._names))
        if len(ids) > 0:
            if self._scaler is not None:
                result = self._scaler.transform(result)
            if self._minmax is not None:
                result = self._minmax.transform(result)
        return dense_result(ids, _BLOCK, result.reshape(len(ids), len(self._names)), self._names)


@register("landmarkGenesReduced", description="Reduced landmark gene set used by DrugGNN and PharmaFormer.")
class LandmarkGenesReducedFeaturizer(LandmarkGenesFeaturizer):
    """Landmark genes reduced featurizer component."""

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
