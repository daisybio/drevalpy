"""Feature loaders for component-based models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.components._feature_dataset import FeatureDataset
from drevalpy.components.data_loading.view_resolution import views_from_featurizer_config
from drevalpy.models.config import ModelConfig, ResolvedModelConfig

if TYPE_CHECKING:
    from drevalpy.datasets.mudataset import MuDataset


def _unwrap_model_config(config: ModelConfig | ResolvedModelConfig) -> tuple[ModelConfig, ResolvedModelConfig | None]:
    if isinstance(config, ResolvedModelConfig):
        return config.template, config
    return config, None


# ---------------------------------------------------------------------------
# MuDataset-based feature construction
# ---------------------------------------------------------------------------


def _build_cell_line_feature_dict(
    mudataset: MuDataset,
    views: list[str],
    cell_line_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the nested ``{entity_id: {view: array}}`` structure from a MuDataset.

    :param mudataset: Dataset wrapper providing cell-line feature accessors.
    :param views: Raw view names required by the featurizer tree.
    :param cell_line_ids: Cell-line IDs to include.
    :returns: Nested dict suitable for ``FeatureDataset``.
    """
    features: dict[str, dict[str, np.ndarray]] = {}
    view_matrices: dict[str, np.ndarray] = {}
    for view in views:
        view_matrices[view] = mudataset.get_cell_line_features(view, cell_line_ids)
    for i, cl_id in enumerate(cell_line_ids):
        features[str(cl_id)] = {view: view_matrices[view][i] for view in views}
    return features


def _build_drug_feature_dict(
    mudataset: MuDataset,
    views: list[str],
    drug_ids: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the nested ``{entity_id: {view: array}}`` structure for drugs.

    :param mudataset: Dataset wrapper providing drug feature accessors.
    :param views: Drug view names (varm keys) required by the featurizer tree.
    :param drug_ids: Drug IDs to include.
    :returns: Nested dict suitable for ``FeatureDataset``.
    """
    features: dict[str, dict[str, np.ndarray]] = {}
    view_matrices: dict[str, np.ndarray] = {}
    for view in views:
        view_matrices[view] = mudataset.get_drug_features(view, drug_ids)
    for i, drug_id in enumerate(drug_ids):
        features[str(drug_id)] = {view: view_matrices[view][i] for view in views}
    return features


def build_cell_line_features_from_mudataset(
    mudataset: MuDataset,
    config: ModelConfig | ResolvedModelConfig,
    cell_line_ids: np.ndarray,
) -> FeatureDataset:
    """Construct a ``FeatureDataset`` for cell lines from a MuDataset.

    Uses the model config to determine which views are needed, then pulls
    data from the MuDataset. For identity/tissue-only featurizers, builds a
    minimal FeatureDataset with just IDs or tissue labels.

    :param mudataset: MuDataset providing all feature data.
    :param config: Model config declaring required views.
    :param cell_line_ids: Cell-line IDs to include in the feature dataset.
    :returns: FeatureDataset populated from the MuDataset.
    """
    template, resolved = _unwrap_model_config(config)
    featurizer = template.cell_line_featurizer

    if featurizer is not None and featurizer.name == "tissue":
        tissues = mudataset.get_tissue(cell_line_ids)
        return FeatureDataset(
            features={str(cl): {"tissue": np.array([t])} for cl, t in zip(cell_line_ids, tissues, strict=True)}
        )

    if template.predictor.name == "naiveMeanEffects" and (featurizer is None or featurizer.name == "identity"):
        tissues = mudataset.get_tissue(cell_line_ids)
        return FeatureDataset(
            features={
                str(cl): {"cell_line_id": np.array([cl]), "tissue": np.array([t])}
                for cl, t in zip(cell_line_ids, tissues, strict=True)
            }
        )

    if featurizer is None or template.cell_line_entity_id_only():
        return FeatureDataset(features={str(cl): {"cell_line_id": np.array([cl])} for cl in cell_line_ids})

    views = views_from_featurizer_config(featurizer, registry="cell_line", resolved=resolved)
    if not views:
        return FeatureDataset(features={str(cl): {"cell_line_id": np.array([cl])} for cl in cell_line_ids})
    feat_dict = _build_cell_line_feature_dict(mudataset, views, cell_line_ids)
    return FeatureDataset(features=feat_dict)


def build_drug_features_from_mudataset(
    mudataset: MuDataset,
    config: ModelConfig | ResolvedModelConfig,
    drug_ids: np.ndarray,
) -> FeatureDataset | None:
    """Construct a ``FeatureDataset`` for drugs from a MuDataset.

    :param mudataset: MuDataset providing all feature data.
    :param config: Model config declaring required drug views.
    :param drug_ids: Drug IDs to include in the feature dataset.
    :returns: FeatureDataset populated from the MuDataset, or None if no drug featurizer.
    """
    template, resolved = _unwrap_model_config(config)
    if template.drug_featurizer is None:
        return None

    if template.drug_entity_id_only():
        return FeatureDataset(features={str(d): {"drug_id": np.array([d])} for d in drug_ids})

    views = views_from_featurizer_config(template.drug_featurizer, registry="drug", resolved=resolved)
    if not views:
        return FeatureDataset(features={str(d): {"drug_id": np.array([d])} for d in drug_ids})
    feat_dict = _build_drug_feature_dict(mudataset, views, drug_ids)
    return FeatureDataset(features=feat_dict)
