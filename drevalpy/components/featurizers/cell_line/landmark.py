"""Landmark gene featurizers for literature models."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


def _load_gene_indices(feature_dataset, view: str, gene_list_stem: str) -> list[int]:
    meta = feature_dataset.meta_info.get(view)
    if meta is None:
        msg = f"FeatureDataset meta_info missing view {view!r}"
        raise ValueError(msg)
    gene_list_path = Path("data/meta/gene_lists") / f"{gene_list_stem}.csv"
    if not gene_list_path.is_file():
        return list(range(len(meta)))
    gene_info = pd.read_csv(gene_list_path)
    if "gene_name" not in gene_info.columns:
        return list(range(len(meta)))
    selected = set(gene_info["gene_name"].astype(str))
    indices: list[int] = []
    for index, gene in enumerate(meta):
        if str(gene) in selected:
            indices.append(index)
    if not indices:
        msg = f"No genes from {gene_list_stem!r} matched view {view!r}"
        raise ValueError(msg)
    return indices


def _subset_matrix(
    features,
    entity_ids: np.ndarray,
    *,
    view: str,
    gene_indices: list[int],
    scaler: StandardScaler | None = None,
    minmax: MinMaxScaler | None = None,
    arcsinh: bool = True,
) -> np.ndarray:
    matrix = stack_view_matrix(features, view, entity_ids).astype(np.float64)
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
    category="general_purpose",
)
class LandmarkGenesFeaturizer(CellLineFeaturizer):
    """Landmark genes featurizer component."""

    output_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="gene_expression")

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        gene_list_stem: str = "landmark_genes",
        standardize: bool = True,
        minmax_scale: bool = False,
    ) -> None:
        self._view = view
        self._gene_list_stem = gene_list_stem
        self._standardize = standardize
        self._minmax_scale = minmax_scale
        self._gene_indices: list[int] = []
        self._scaler: StandardScaler | None = None
        self._minmax: MinMaxScaler | None = None
        self._output_dim = 0

    def fit(self, features, *, entity_ids: np.ndarray | None = None) -> LandmarkGenesFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        self._gene_indices = _load_gene_indices(features, self._view, self._gene_list_stem)
        self._output_dim = len(self._gene_indices)
        matrix = stack_view_matrix(features, self._view, ids).astype(np.float64)[:, self._gene_indices]
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
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
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return _subset_matrix(
            features,
            entity_ids,
            view=self._view,
            gene_indices=self._gene_indices,
            scaler=self._scaler,
            minmax=self._minmax,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim


@register_cell_line_featurizer(
    "landmarkGenesReduced",
    description="Reduced landmark gene set used by DrugGNN and PharmaFormer.",
    category="general_purpose",
)
class LandmarkGenesReducedFeaturizer(LandmarkGenesFeaturizer):
    """Landmark genes reduced featurizer component."""

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        standardize: bool = False,
        minmax_scale: bool = False,
    ) -> None:
        super().__init__(
            view=view,
            gene_list_stem="landmark_genes_reduced",
            standardize=standardize,
            minmax_scale=minmax_scale,
        )
