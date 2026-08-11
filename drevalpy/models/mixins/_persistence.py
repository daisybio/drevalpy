"""Checkpoint persistence mixin for DRPModel subclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from upath import UPath as Path

from drevalpy.models.mixins._persistence_io import (
    CorruptedCheckpointError,
    IncompatibleModelCheckpointError,
    load_model_payload,
    save_model,
)

if TYPE_CHECKING:
    from drevalpy.models.drp_model import DRPModel


class DRPPersistenceMixin:
    """Mixin providing save/load checkpoint operations for DRPModel subclasses.

    Delegates low-level archive I/O to ``drevalpy.models.mixins._persistence_io``.
    """

    def save(self, path: str | Path) -> None:
        """Persist model identity, config, and fitted component state.

        :param path: Archive file path; ``.zip`` is appended when missing.
        """
        save_model(self, path)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> DRPModel:
        """Load a fitted model checkpoint into a new instance of this class.

        :param path: Archive file path; ``.zip`` is appended when missing.
        :returns: Fitted model instance with restored component state.
        :raises IncompatibleModelCheckpointError: If the stored model name does not match this class.
        :raises CorruptedCheckpointError: If the archive payload is invalid or incomplete.
        """
        model_name, config, state = load_model_payload(path)
        if model_name != cls.get_model_name():  # type: ignore[attr-defined]
            raise IncompatibleModelCheckpointError(
                f"checkpoint model_name {model_name!r} does not match {cls.get_model_name()!r}"  # type: ignore[attr-defined]
            )
        instance: Any = cls._from_resolved_config(config)  # type: ignore[attr-defined]
        if instance._stack is None:
            raise CorruptedCheckpointError("failed to materialize component stack from checkpoint")
        try:
            instance._stack.restore_component_state(state)
        except (ValueError, RuntimeError) as exc:
            raise CorruptedCheckpointError(
                f"checkpoint component state is invalid: {exc}" if str(exc) else "checkpoint component state is invalid"
            ) from exc
        if not instance._stack.is_fitted():
            raise CorruptedCheckpointError("checkpoint did not restore a fitted predictor")
        instance._empty_training = False
        return instance
