"""The no-op checkpoint hooks every component kind inherits."""

from __future__ import annotations


class CheckpointStateMixin:
    """Persist fitted state through ``get_state`` / ``set_state``.

    ``Featurizer`` and ``Predictor`` share one checkpoint protocol. The defaults are
    its no-op end: a component with nothing fitted returns an empty state.
    Subclasses that hold fitted state override both methods so a checkpoint round-trips it.
    """

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for checkpoint persistence.

        :returns: JSON-serializable mapping of fitted attributes.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        _ = state
