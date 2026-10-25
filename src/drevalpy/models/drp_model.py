"""The public model runtime: one named recipe and its complete fitted pipeline.

Fitted state lives in ``drevalpy/models/_pipeline.py``, whose docstring lists the module of each
concern.
"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath

from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.models._pipeline import _ModelPipeline
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.split_mask import SplitMask

if TYPE_CHECKING:
    from drevalpy.models.config.resolved import ResolvedModelConfig
    from drevalpy.models.response_transformation import ResponseTransformer


class DRPModel:
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
        response_transformation: ResponseTransformer | None = None,
    ) -> None:
        """Configure a fresh model without training any components.

        :param spec: Named preset or custom component configuration.
        :param hyperparameters: Optional public component parameter overrides.
        :param response_transformation: Cloneable, invertible response transformation prototype.
        :raises TypeError: If spec is not a ModelSpec.
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError("DRPModel requires a ModelSpec; use ModelSpec.from_name(name) for a preset")
        self._spec = spec
        self._pipeline = _ModelPipeline(spec.resolve(hyperparameters), response_transformation)

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
    def sample_views(self) -> list[str]:
        """Raw sample views consumed by the resolved model."""
        return self.config.sample_views()

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
        model_checkpoint_dir: str | UPath | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Start a fresh fit using training and optional disjoint stopping data.

        :param dataset: Response matrix and raw features.
        :param scope: Training response positions.
        :param early_stopping_scope: Disjoint positions used only for stopping.
        :param model_checkpoint_dir: Directory for component training checkpoints. When omitted, a temporary
            directory is used for this fit and removed afterwards.
        :param n_jobs: Number of worker processes a single-drug predictor fits its per-drug models in; ``-1`` uses
            all CPUs. It changes the time a fit takes, not its result, and is ignored by every other model.
        """
        check_n_jobs(n_jobs)
        self._pipeline.reset()
        directory = (
            tempfile.TemporaryDirectory(prefix="drevalpy-checkpoints-")
            if model_checkpoint_dir is None
            else nullcontext(model_checkpoint_dir)
        )
        with directory as checkpoint_dir:
            context = TrainingContext(checkpoint_dir=UPath(checkpoint_dir), n_jobs=n_jobs)
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
        """Restore a complete fitted model, including its response transformation.

        .. warning::
           Checkpoints are unpickled (``joblib``, and ``torch.load`` with
           ``weights_only=False`` for neural weights), which can execute arbitrary
           code. Load checkpoints only from trusted sources.

        :param path: Checkpoint archive path.
        :returns: Restored concrete model.
        """
        from drevalpy.models.mixins._persistence_io import load_model

        return load_model(path)
