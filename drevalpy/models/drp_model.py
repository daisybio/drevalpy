"""The public model runtime: one named recipe and its complete fitted pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.models._pipeline import _ModelPipeline
from drevalpy.models.mixins._logging import _DRPLoggingMixin
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.split_mask import SplitMask

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

    from drevalpy.models.config.resolved import ResolvedModelConfig


class DRPModel(_DRPLoggingMixin):
    """Train, predict, and persist a model built from an immutable ``ModelSpec``.

    Predictions always have one value per requested response position, in the
    dataset's original response units. The private pipeline owns all fitted
    state, including feature preprocessing and response transformation.
    """

    def __init__(
        self,
        spec: ModelSpec,
        hyperparameters: dict[str, Any] | None = None,
        *,
        response_transformation: TransformerMixin | None = None,
    ) -> None:
        """Configure a fresh model without training any components.

        :param spec: Named preset or custom component configuration.
        :param hyperparameters: Optional public component parameter overrides.
        :param response_transformation: Cloneable, invertible target transformer prototype.
        :raises TypeError: If spec is not a ModelSpec.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError("DRPModel requires a ModelSpec; use ModelSpec.from_name(name) for a preset")
        self._spec = spec
        self._pipeline = _ModelPipeline(spec.resolve(hyperparameters), response_transformation)
        self.wandb_project: str | None = None
        self.wandb_run: Any = None
        self.wandb_config: dict[str, Any] | None = None
        self._in_hyperparameter_tuning = False

    @property
    def spec(self) -> ModelSpec:
        """Unfitted specification bound to this pipeline for its entire lifetime."""
        return self._spec

    @property
    def name(self) -> str:
        """Public name used by results, checkpoints, and logging."""
        return self.spec.name

    @property
    def config(self) -> ResolvedModelConfig:
        """Immutable configuration with concrete component parameters."""
        return self._pipeline.resolved

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Independent public parameter mapping for this model."""
        return public_hyperparameters_from_config(self.config)

    @property
    def cell_line_views(self) -> list[str]:
        """Raw cell-line views consumed by the resolved model."""
        return self.config.cell_line_views()

    @property
    def drug_views(self) -> list[str]:
        """Raw drug views consumed by the resolved model."""
        return self.config.drug_views()

    def train(
        self,
        dataset: Dataset,
        scope: SplitMask,
        *,
        early_stopping_scope: SplitMask | None = None,
        model_checkpoint_dir: str | UPath = "checkpoints",
    ) -> None:
        """Start a fresh fit using training and optional disjoint stopping data.

        :param dataset: Response matrix and raw features.
        :param scope: Training response positions.
        :param early_stopping_scope: Disjoint positions used only for stopping.
        :param model_checkpoint_dir: Directory for component training checkpoints.
        """
        self._pipeline.reset()
        context = TrainingContext(
            checkpoint_dir=UPath(model_checkpoint_dir), logging_metadata={"model_name": self.name}
        )
        self._pipeline.fit(dataset, scope, early_stopping_scope, training_context=context)

    def predict(self, dataset: Dataset, scope: SplitMask) -> np.ndarray:
        """Predict original-unit responses in scope order, preserving missing positions.

        :param dataset: Response matrix and raw features.
        :param scope: Requested response positions, including unobserved responses.
        :returns: One prediction or NaN per position.
        """
        return self._pipeline.predict(dataset, scope)

    def save(self, path: str | UPath) -> None:
        """Save the complete fitted pipeline as one versioned archive.

        :param path: Checkpoint path; .zip is appended when needed.
        """
        from drevalpy.models.mixins._persistence_io import save_model

        save_model(self, path)

    @classmethod
    def load(cls, path: str | UPath) -> DRPModel:
        """Restore a complete fitted model, including its target transformer.

        :param path: Checkpoint archive path.
        :returns: Restored concrete model.
        """
        from drevalpy.models.mixins._persistence_io import load_model

        return load_model(path)
