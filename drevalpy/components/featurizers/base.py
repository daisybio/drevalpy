"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.datasets.dataset import FeatureDataset


class Featurizer(ABC):
    """Transform feature tables into per-entity representation payloads.

    Cell-line featurizers consume cell-line features; drug featurizers consume
    drug features. Subclasses must be registered
    to the cell-line or drug featurizer registry using
    ``@register_cell_line_featurizer`` or ``@register_drug_featurizer``, so that
    they can be discovered and used in models.
    """

    contract: ClassVar[FeatureContract]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject class-body ``contract`` assignments; registration sets it later.

        :param kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If ``contract`` is assigned on the subclass body.
        """
        super().__init_subclass__(**kwargs)
        if "contract" in cls.__dict__:
            msg = (
                f"{cls.__name__}: do not set contract on the class body; "
                "pass contract= to @register_cell_line_featurizer / @register_drug_featurizer"
            )
            raise TypeError(msg)

    @abstractmethod
    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> Featurizer:
        """Fit on the entities given by *entity_ids* (or all entities when ``None``).

        :param features: Raw feature views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param context: Optional training context shared across featurizers.

        :returns: Fitted featurizer instance (usually ``self``).
        """

    @abstractmethod
    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return one payload row per entity id in *entity_ids*.

        :param features: Raw feature views for the entity type.
        :param entity_ids: Entity identifiers to transform.

        :returns: Feature payloads aligned with *entity_ids*.
        """

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Return named feature blocks; default is a single ``default`` block.

        :param features: Raw feature views for the entity type.
        :param entity_ids: Entity identifiers to transform.

        :returns: Mapping of block name to ``FeatureBlock`` payloads aligned with *entity_ids*.
        """
        return {
            "default": numeric_feature_block(self.transform(features, entity_ids)),
        }

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after ``fit``.

        :returns: Result.
        """

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO.

        :returns: Mapping of parameter name to Ray Tune-style spec dicts.
        """
        return {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space.

        :returns: Parameter names mapped to their declared ``default`` values.
        """
        return {
            key: spec["default"]
            for key, spec in cls.get_hyperparameter_space().items()
            if isinstance(spec, dict) and "default" in spec
        }

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load the raw dataset required by this featurizer.

        Featurizers that require bespoke on-disk artifacts override this hook.
        Generic views continue to be loaded by the model data-loading layer.

        :param data_path: Parent directory for dataset artifacts.
        :param dataset_name: Dataset folder name (for example ``"GDSC1"``).
        :param kwargs: Featurizer-specific loader options from the model config.
        :raises NotImplementedError: When the featurizer does not provide a custom loader.
        """
        _ = data_path, dataset_name, kwargs
        raise NotImplementedError

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges.

        :returns: JSON-serializable mapping of fitted attributes.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        _ = state
