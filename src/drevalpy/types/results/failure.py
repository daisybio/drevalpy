"""Record of a run that raised instead of producing a result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from upath import UPath as Path

from drevalpy.errors import DrEvalError
from drevalpy.types.enums.randomization import randomization_label
from drevalpy.types.results._identity import RunIdentity

if TYPE_CHECKING:
    from pydantic import TypeAdapter

#: File name, inside an experiment directory, of the recorded failures.
FAILURES_FILE = "failures.json"


@dataclass(frozen=True)
class RunFailure:
    """One run of an experiment that failed, kept so the other runs can finish.

    :ivar identity: Identity of the run that failed. Its provenance is empty, since the run never got as far as
        recording any.
    :ivar error_type: Class name of the exception.
    :ivar message: The exception's message.
    """

    identity: RunIdentity
    error_type: str
    message: str

    @classmethod
    def from_exception(cls, exc: BaseException, identity: RunIdentity) -> RunFailure:
        """Describe *exc* as a failure of one run.

        :param exc: The exception the run raised.
        :param identity: Identity of the run that raised.
        :returns: The failure record.
        """
        return cls(identity=identity, error_type=type(exc).__name__, message=str(exc))

    def __str__(self) -> str:
        """One-line description for logs and tables."""
        identity = self.identity
        extras = [f"randomization {randomization_label(identity.randomization)}"] if identity.randomization else []
        if identity.robustness_repeat is not None:
            extras.append(f"robustness repeat {identity.robustness_repeat}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{identity.model_name} fold {identity.fold_index}{suffix}: {self.error_type}: {self.message}"


def _failures_adapter() -> TypeAdapter[list[RunFailure]]:
    from pydantic import TypeAdapter

    return TypeAdapter(list[RunFailure])


def write_failures(directory: Path, failures: Sequence[RunFailure]) -> None:
    """Write ``failures.json`` into *directory*, or nothing when there are no failures.

    :param directory: Existing experiment directory.
    :param failures: The failures to record.
    """
    if failures:
        (directory / FAILURES_FILE).write_bytes(_failures_adapter().dump_json(list(failures), indent=2))


def read_failures(directory: Path) -> list[RunFailure]:
    """Read the failures :func:`write_failures` stored in *directory*.

    :param directory: Experiment directory.
    :returns: The recorded failures, empty when the directory has no ``failures.json``.
    """
    path = directory / FAILURES_FILE
    return _failures_adapter().validate_json(path.read_bytes()) if path.exists() else []


class ExperimentFailedError(DrEvalError, RuntimeError):
    """Raised by ``run`` when every run of an experiment failed."""

    def __init__(self, failures: tuple[RunFailure, ...] | list[RunFailure]) -> None:
        """Create the error from the failures that left no result.

        :param failures: The recorded failures, one per run; kept as the ``failures`` attribute.
        """
        self.failures: tuple[RunFailure, ...] = tuple(failures)
        super().__init__(f"{len(self.failures)} run(s) failed")

    def __str__(self) -> str:
        """List every failure under a count, one indented line each.

        :returns: ``N run(s) failed:`` followed by one ``  <failure>`` line per failure.
        """
        lines = [f"{len(self.failures)} run(s) failed:"]
        lines.extend(f"  {failure}" for failure in self.failures)
        return "\n".join(lines)
