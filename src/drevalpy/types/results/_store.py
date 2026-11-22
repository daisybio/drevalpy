"""The on-disk layout of an experiment, and the store ``drevalpy.experiment`` saves each run into as it finishes.

An experiment directory holds ``metadata.json``, ``failures.json`` when a run failed, and one directory
per model, named by :func:`~drevalpy._utils.filenames.unique_safe_filenames` of ``metadata["models"]``.
A model directory holds one ``.npz`` per run, named by :func:`run_filename` after the run's
:attr:`~drevalpy.types.results._identity.RunIdentity.checkpoint_key`. The key leaves out the provenance,
so ``run`` can look a run up before fitting it, and the name does not depend on the order the runs
finished in. :meth:`ExperimentResult.save <drevalpy.types.results.ExperimentResult.save>` and
:class:`RunStore` write this one layout, and ``ExperimentResult.load`` reads both.

:class:`RunStore` writes ``metadata.json`` with the run's ``settings`` as soon as it opens a directory,
so a killed run can be resumed: a later ``run`` with equal settings loads every run that is on disk
instead of fitting it again. A directory with runs and other settings (or none, as ``ExperimentResult.save``
writes) is refused unless ``overwrite`` deletes it. ``overwrite`` deletes only the files of this layout, so a
directory that holds files but no ``metadata.json`` of an experiment is always refused. Failures are not saved
per run, so a resumed run retries them; :meth:`RunStore.finish` writes ``failures.json`` once, at the end.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from upath import UPath as Path

from drevalpy._utils.filenames import safe_filename, unique_safe_filenames
from drevalpy.errors import InvalidInputError, OutputDirectoryConflictError
from drevalpy.log import get_logger
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results.failure import FAILURES_FILE, RunFailure, write_failures
from drevalpy.types.results.run import RunResult

logger = get_logger(__name__)

#: File name, inside an experiment directory, of the model order, the normalization and the run settings.
METADATA_FILE = "metadata.json"

# 64 bits of the key digest: unique across the runs of any experiment, short enough for Windows paths.
_DIGEST_LENGTH = 16


def run_filename(identity: RunIdentity) -> str:
    """Name the file of a run: a readable label of its fold and stress test, then a digest of its checkpoint key.

    :param identity: Identity of the run, before or after it was fitted.
    :returns: A file name such as ``fold-0_SVRC_gene_expression_repeat-1_<digest>.npz``.
    """
    key = json.dumps(identity.checkpoint_key, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    label = [f"fold-{identity.fold_index}", *(identity.randomization or ())]
    if identity.robustness_repeat is not None:
        label.append(f"repeat-{identity.robustness_repeat}")
    return f"{safe_filename('_'.join(label))}_{digest}.npz"


def save_runs(directory: Path, runs: Sequence[RunResult]) -> None:
    """Write *runs* into a model directory, and remove the run files of other runs an earlier save left there.

    :param directory: Model directory; created when missing.
    :param runs: Runs of one model.
    :raises InvalidInputError: If two runs have the same checkpoint key, so one would overwrite the other.
    """
    names = [run_filename(run.identity) for run in runs]
    if len(set(names)) != len(names):
        raise InvalidInputError(f"Two runs of {runs[0].model_name} have the same fold and stress test")
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.npz"):
        if stale.name not in names:
            stale.unlink()
    for run, name in zip(runs, names, strict=True):
        run.save(directory / name)


def load_runs(directory: Path, *, with_trials: bool) -> list[RunResult]:
    """Read every run file of a model directory, ordered by fold, robustness repeat and randomization.

    :param directory: Model directory.
    :param with_trials: Forwarded to :meth:`RunResult.load`.
    :returns: The runs, empty when the directory does not exist or holds no run file.
    """
    files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
    return sorted((RunResult.load(file, with_trials=with_trials) for file in files), key=_run_order)


def _run_order(run: RunResult) -> tuple:
    """Order runs as ``run`` makes them: by fold, the original fold before its repeats, the original data first."""
    repeat = -1 if run.robustness_repeat is None else run.robustness_repeat
    return (run.fold_index, run.fold_id, repeat, run.randomization is not None, run.randomization or ("", ""))


def write_metadata(
    directory: Path,
    *,
    models: Sequence[str],
    normalized_by: str | None,
    failures: Sequence[RunFailure],
    settings: Mapping[str, Any] | None = None,
) -> None:
    """Write ``metadata.json`` and ``failures.json``, removing a ``failures.json`` that no failure is left for.

    ``metadata.json`` is renamed into place once written, so a kill never leaves a half-written one.

    :param directory: Existing experiment directory.
    :param models: Model names, in the order their directories are named.
    :param normalized_by: Reference model of a normalized experiment, else ``None``.
    :param failures: Recorded failures.
    :param settings: Settings of the ``run`` that made the experiment; ``None`` for an experiment made otherwise.
    """
    meta: dict[str, Any] = {"normalized_by": normalized_by, "models": list(models)}
    if settings is not None:
        meta["settings"] = settings
    partial = directory / f"{METADATA_FILE}.partial"
    partial.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.replace(directory / METADATA_FILE)
    (directory / FAILURES_FILE).unlink(missing_ok=True)
    write_failures(directory, failures)


def read_metadata(directory: Path) -> dict[str, Any]:
    """Read ``metadata.json`` of an experiment directory.

    :param directory: Experiment directory.
    :returns: The ``normalized_by`` and ``models`` entries, and ``settings`` when ``run`` wrote the directory.
    """
    return json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))


class RunStore:
    """An experiment directory that ``run`` saves each run into when it finishes and loads finished runs from."""

    def __init__(self, directory: Path, settings: dict[str, Any]) -> None:
        """Wrap a directory that :meth:`open` checked; use :meth:`open` instead.

        :param directory: Experiment directory.
        :param settings: JSON run settings, with the model names under ``settings["models"]``.
        """
        self._directory = directory
        self._settings = settings
        self._models = [model["name"] for model in settings["models"]]
        self._model_directories = unique_safe_filenames(self._models)

    @classmethod
    def open(cls, directory: str | Path, settings: dict[str, Any], *, overwrite: bool) -> RunStore:
        """Open an experiment directory for a run: start it when it holds no runs, resume it when it is the same run.

        :param directory: Output directory of the run.
        :param settings: JSON settings that decide the results of the run, with ``"models"`` a list of objects that
            each have a ``"name"``; recorded in ``metadata.json`` and compared with the recorded ones as they are.
        :param overwrite: Delete the experiment in the directory first: its ``metadata.json``, ``failures.json`` and
            model directories. Other files in the directory stay.
        :returns: The store.
        :raises OutputDirectoryConflictError: If the directory holds files but no experiment (whatever *overwrite*
            is), or, unless *overwrite* is set, runs and other settings than *settings*.
        """
        path = Path(directory)
        meta = _experiment_metadata(path)
        if meta is None and path.exists() and any(path.iterdir()):
            raise OutputDirectoryConflictError(
                f"The output directory {path} holds files that no drevalpy run wrote. Choose an empty or new directory."
            )
        if meta is not None and overwrite:
            logger.info("Deleting the experiment in %s, since overwrite was asked for", path)
            _delete_experiment(path, meta["models"])
        elif meta is not None and any(entry.name not in (METADATA_FILE, FAILURES_FILE) for entry in path.iterdir()):
            if meta.get("settings") != settings:
                raise OutputDirectoryConflictError(
                    f"The output directory {path} holds results of a run with other settings. Pass --overwrite "
                    "(overwrite=True in Python) to delete them, or choose another output directory."
                )
            logger.info("Resuming the run in %s: runs that are already saved are loaded, not fitted", path)
        path.mkdir(parents=True, exist_ok=True)
        store = cls(path, settings)
        store.finish(())
        return store

    def _file(self, identity: RunIdentity) -> Path:
        return self._directory / self._model_directories[identity.model_name] / run_filename(identity)

    def load(self, identity: RunIdentity) -> RunResult | None:
        """Read the saved run with *identity*'s checkpoint key.

        :param identity: Identity of the run about to be fitted; its provenance is ignored.
        :returns: The saved run, or ``None`` when it is not on disk.
        """
        path = self._file(identity)
        if not path.exists():
            return None
        logger.debug("Loading the saved run %s", path)
        return RunResult.load(path)

    def save(self, run: RunResult) -> None:
        """Write one finished run; it is renamed into place only when it is complete, so a kill leaves no broken file.

        :param run: The run, of one of the models in the settings.
        """
        path = self._file(run.identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.partial")
        run.save(partial)
        partial.replace(path)

    def finish(self, failures: Sequence[RunFailure]) -> None:
        """Write ``metadata.json`` with the settings, and ``failures.json`` with *failures* (removed when empty).

        :param failures: The failures of the run; may be every run of it.
        """
        write_metadata(
            self._directory, models=self._models, normalized_by=None, failures=failures, settings=self._settings
        )


def _experiment_metadata(directory: Path) -> dict[str, Any] | None:
    """Return the ``metadata.json`` of the experiment in *directory*, or ``None`` when it holds none."""
    try:
        meta = read_metadata(directory)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and isinstance(meta.get("models"), list) else None


def _delete_experiment(directory: Path, models: list[str]) -> None:
    """Delete the files of the experiment layout in *directory*, and nothing else."""
    for name in (METADATA_FILE, FAILURES_FILE):
        (directory / name).unlink(missing_ok=True)
    for name in unique_safe_filenames(models).values():
        model_directory = directory / name
        if model_directory.exists():
            directory.fs.rm(model_directory.path, recursive=True)
