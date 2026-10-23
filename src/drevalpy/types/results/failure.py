"""Record of a run that raised instead of producing a result."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from upath import UPath as Path

from drevalpy.errors import DrEvalError

#: File name, inside an experiment directory, of the recorded failures.
FAILURES_FILE = "failures.json"


@dataclass(frozen=True)
class RunFailure:
    """One run of an experiment that failed, kept so the other runs can finish.

    Attributes:
        model_name: Name of the model whose run failed.
        fold_index: Index of the fold the run was on.
        randomization: ``(mode, view)`` of the randomized dataset the run used,
            as on :class:`~drevalpy.types.results.run.RunResult`, or ``None`` for
            the original dataset.
        robustness_trial: Index of the robustness repeat, or ``None`` for an ordinary run.
        error_type: Class name of the exception.
        message: The exception's message.
    """

    model_name: str
    fold_index: int
    randomization: tuple[str, str] | None
    robustness_trial: int | None
    error_type: str
    message: str

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        model_name: str,
        fold_index: int,
        randomization: tuple[str, str] | None = None,
        robustness_trial: int | None = None,
    ) -> RunFailure:
        """Describe *exc* as a failure of one run.

        Args:
            exc: The exception the run raised.
            model_name: Name of the model whose run failed.
            fold_index: Index of the fold the run was on.
            randomization: ``(mode, view)`` of the randomized dataset, if any.
            robustness_trial: Index of the robustness repeat, if any.

        Returns:
            The failure record.
        """
        return cls(
            model_name=model_name,
            fold_index=fold_index,
            randomization=randomization,
            robustness_trial=robustness_trial,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the failure as a JSON-safe dict (``randomization`` as a list)."""
        data = asdict(self)
        data["randomization"] = list(self.randomization) if self.randomization else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunFailure:
        """Rebuild a failure from :meth:`to_dict` output.

        Args:
            data: Mapping with the six fields; ``randomization`` and ``robustness_trial`` may be absent.

        Returns:
            The failure record.
        """
        return cls(
            model_name=data["model_name"],
            fold_index=int(data["fold_index"]),
            randomization=(str(data["randomization"][0]), str(data["randomization"][1]))
            if data.get("randomization")
            else None,
            robustness_trial=None if data.get("robustness_trial") is None else int(data["robustness_trial"]),
            error_type=data["error_type"],
            message=data["message"],
        )

    def __str__(self) -> str:
        """One-line description for logs and tables."""
        extras = [f"randomization {self.randomization[0]} ({self.randomization[1]})"] if self.randomization else []
        if self.robustness_trial is not None:
            extras.append(f"robustness trial {self.robustness_trial}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{self.model_name} fold {self.fold_index}{suffix}: {self.error_type}: {self.message}"


def write_failures(directory: Path, failures: Sequence[RunFailure]) -> None:
    """Write ``failures.json`` into *directory*, or nothing when there are no failures.

    Args:
        directory: Existing experiment directory.
        failures: The failures to record.
    """
    if failures:
        (directory / FAILURES_FILE).write_text(json.dumps([failure.to_dict() for failure in failures], indent=2))


def read_failures(directory: Path) -> list[RunFailure]:
    """Read the failures :func:`write_failures` stored in *directory*.

    Args:
        directory: Experiment directory.

    Returns:
        The recorded failures, empty when the directory has no ``failures.json``.
    """
    path = directory / FAILURES_FILE
    return [RunFailure.from_dict(item) for item in json.loads(path.read_text())] if path.exists() else []


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
