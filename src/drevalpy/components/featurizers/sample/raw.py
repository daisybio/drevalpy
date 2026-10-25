"""Generic raw dense pass-through featurizer for one omics view."""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "raw",
    description="Pass through one dense omics view without preprocessing.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class RawSampleFeaturizer(DenseViewSampleFeaturizer):
    """Featurize one omics view as a dense matrix without transformation."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str, gene_list: str | None = None) -> None:
        """Bind the omics view to pass through.

        :param view: Name of the omics view to emit unchanged.
        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every gene of
            it. ``None`` keeps the whole view.
        :raises ValueError: If *view* is empty or blank.
        """
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        super().__init__(view=view)
        self._gene_list = gene_list

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Bind the pass-through to the genes of the gene list, or to the whole view without one.

        :param source: Feature source providing view matrices.
        :returns: Gene names in gene-list order, or every feature name of the view.
        """
        if self._gene_list is None:
            return super()._names_to_fit(source)
        return select_genes(source, self._view, self._gene_list)

    def _fetch_hyperparameters(self) -> dict[str, Any] | None:
        """Match a stored variant only when it was built for the same gene list.

        :returns: The gene-list setting, or ``None`` for the whole view.
        """
        return None if self._gene_list is None else {"gene_list": self._gene_list}
