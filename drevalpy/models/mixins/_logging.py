"""Weights & Biases and metric helpers mixed into DRPModel.

``wandb`` is imported inside the methods that use it. ``DRPModel`` is on the
registration path of ``import drevalpy``, and importing ``wandb`` costs ~0.11s
even for the overwhelming majority of runs that never enable it. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def _wandb() -> ModuleType:
    """Import and return the ``wandb`` module.

    Every ``wandb`` reference in this file goes through here, so the dependency
    is paid for on first use rather than at ``import drevalpy`` time, and so
    tests have a single seam to patch.

    :returns: The imported ``wandb`` module.
    """
    import wandb

    return wandb


class _DRPLoggingMixin:
    """Shared wandb / evaluation helpers for concrete DRPModel instances."""

    wandb_project: str | None
    wandb_run: Any
    wandb_config: dict[str, Any] | None
    _in_hyperparameter_tuning: bool

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model identity; implemented by ``DRPModel``.

        :raises NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Return instance hyperparameters; implemented by ``DRPModel``.

        :raises NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    def init_wandb(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        finish_previous: bool = True,
    ) -> None:
        """Initialize wandb logging for this model instance.

        :param project: Weights & Biases project name.
        :param config: Optional run configuration dict.
        :param name: Optional run display name; defaults to the model name.
        :param tags: Optional run tags.
        :param finish_previous: Finish any active wandb run before starting a new one.
        """
        self.wandb_project = project
        run_config = dict(config or {})
        if self.hyperparameters and "hyperparameters" not in run_config:
            run_config["hyperparameters"] = self.hyperparameters
        self.wandb_config = run_config

        wandb = _wandb()
        if finish_previous:
            wandb.finish()

        run_name = name or self.get_model_name()
        wandb.init(
            project=project,
            config=self.wandb_config,
            name=run_name,
            tags=tags,
        )
        self.wandb_run = wandb.run

        with suppress(Exception):
            wandb.define_metric("epoch", summary="max")
            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("val_loss", summary="min")
            wandb.define_metric("train_R^2", summary="max")
            wandb.define_metric("val_R^2", summary="max")
            wandb.define_metric("train_Pearson", summary="max")
            wandb.define_metric("val_Pearson", summary="max")

    def is_wandb_enabled(self) -> bool:
        """Return whether wandb logging is active for this instance.

        :returns: ``True`` when a wandb project and run are active.
        """
        return self.wandb_project is not None and (self.wandb_run is not None or _wandb().run is not None)

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        """Store final metrics in the wandb run summary.

        :param metrics: Final scalar metrics to persist in the run summary.
        """
        wandb = _wandb()
        if not self.is_wandb_enabled() or wandb.run is None:
            return
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    def finish_wandb(self) -> None:
        """Finish the wandb run for this model instance."""
        if not self.is_wandb_enabled():
            return
        _wandb().finish()
        self.wandb_run = None
