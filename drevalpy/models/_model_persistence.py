"""Versioned persistence for concrete DRPModel instances.

Checkpoints are a single ZIP archive written atomically to an archive file path.
Callers must only load artifacts they created with ``save_model`` in the same
drevalpy version family.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from drevalpy.models.config import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig

if TYPE_CHECKING:
    from drevalpy.models.drp_model import DRPModel

FORMAT_NAME = "drevalpy-model"
FORMAT_VERSION = 2
PAYLOAD_MEMBER = "payload.joblib"


class ModelCheckpointError(Exception):
    """Base error for DRPModel checkpoint problems."""


class UnsupportedCheckpointFormatError(ModelCheckpointError, ValueError):
    """Raised when checkpoint format or version is not supported."""


class CorruptedCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint payload structure or content is invalid."""


class IncompatibleModelCheckpointError(ModelCheckpointError, ValueError):
    """Raised when checkpoint model identity does not match the loader class."""


def _as_path(path: str | Path) -> Path:
    """Normalize a user path to ``Path``, rejecting trailing separators.

    :param path: Checkpoint archive path string or ``Path``.
    :returns: Normalized path without a trailing directory separator.
    :raises ValueError: If ``path`` ends with a directory separator.
    """
    if isinstance(path, str) and path.endswith(("/", "\\")):
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)
    return Path(path)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Return the archive file path, appending ``.zip`` when missing.

    :param path: Checkpoint archive path; ``.zip`` is appended when missing.
    :returns: Normalized archive ``Path``.
    """
    target = _as_path(path)
    if target.name.lower().endswith(".zip"):
        return target
    return target.with_name(f"{target.name}.zip")


def _reject_directory_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        msg = f"Checkpoint path must be an archive file path, not a directory: {path}"
        raise ValueError(msg)


def _write_archive_atomically(archive_path: Path, payload: dict[str, Any]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{archive_path.name}.", suffix=".tmp", dir=archive_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        buffer = io.BytesIO()
        joblib.dump(payload, buffer)
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
        os.replace(tmp_path, archive_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_payload_from_archive(archive_path: Path) -> Any:
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            try:
                info = archive.getinfo(PAYLOAD_MEMBER)
            except KeyError as exc:
                raise CorruptedCheckpointError(
                    f"checkpoint archive {archive_path} is missing {PAYLOAD_MEMBER!r}"
                ) from exc
            with archive.open(info) as handle:
                return joblib.load(handle)
    except zipfile.BadZipFile as exc:
        raise CorruptedCheckpointError(f"checkpoint archive {archive_path} is not a valid zip file") from exc
    except CorruptedCheckpointError:
        raise
    except Exception as exc:
        raise CorruptedCheckpointError(f"Failed to deserialize checkpoint {archive_path}: {exc}") from exc


def save_model(model: DRPModel, path: str | Path) -> None:
    """Save model identity, config, and component state as one ZIP archive.

    :param model: Trained ``DRPModel`` instance to persist.
    :param path: Archive file path; ``.zip`` is appended when missing.
    :raises RuntimeError: If the model is not trained or lacks a ``ModelConfig``.
    """
    stack = model._stack
    if stack is None or not stack.is_fitted():
        raise RuntimeError("Cannot save: component stack is not trained")
    config = model._resolved_model_config
    if config is None:
        raise RuntimeError("Cannot save a model without its ResolvedModelConfig")
    target = _as_path(path)
    _reject_directory_path(target)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model.get_model_name(),
        "config": config.model_dump(mode="json"),
        "state": stack.component_state(),
    }
    _write_archive_atomically(resolve_checkpoint_path(target), payload)


def _resolved_config_from_checkpoint_payload(payload: dict[str, Any]) -> ResolvedModelConfig:
    version = payload.get("version")
    if version == 2:
        return ResolvedModelConfig.model_validate(payload["config"])
    if version == 1:
        legacy = payload["config"]
        if not isinstance(legacy, dict):
            raise CorruptedCheckpointError("legacy config is not a mapping")
        template_payload = {
            "cell_line_featurizer": _strip_legacy_hyperparameters(legacy.get("cell_line_featurizer")),
            "drug_featurizer": _strip_legacy_hyperparameters(legacy.get("drug_featurizer")),
            "predictor": _strip_legacy_hyperparameters(legacy.get("predictor")),
            "prediction_mode": legacy.get("prediction_mode", "regression"),
        }
        template = ModelConfig.model_validate(template_payload)
        from drevalpy.components.tuning.search_space import resolve_model_config

        return resolve_model_config(template, _legacy_concrete_overrides(legacy, template))
    raise UnsupportedCheckpointFormatError(
        f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}"
    )


def load_model_payload(path: str | Path) -> tuple[str, ResolvedModelConfig, dict[str, object]]:
    """Load and validate a ``DRPModel`` checkpoint payload from an archive path.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :returns: Tuple of ``(model_name, resolved_config, component_state)``.
    :raises FileNotFoundError: If the archive does not exist.
    :raises UnsupportedCheckpointFormatError: If the format or version is unsupported.
    :raises CorruptedCheckpointError: If the payload structure is invalid.
    """
    target = _as_path(path)
    _reject_directory_path(target)
    archive_path = resolve_checkpoint_path(target)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {archive_path}")

    payload = _read_payload_from_archive(archive_path)
    if not isinstance(payload, dict):
        raise CorruptedCheckpointError("checkpoint payload is not a mapping")
    if payload.get("format") != FORMAT_NAME:
        raise UnsupportedCheckpointFormatError(
            f"unsupported checkpoint format/version: {payload.get('format')!r}/{payload.get('version')!r}"
        )
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise CorruptedCheckpointError("checkpoint model_name is missing or invalid")
    try:
        config = _resolved_config_from_checkpoint_payload(payload)
    except UnsupportedCheckpointFormatError:
        raise
    except Exception as exc:
        raise CorruptedCheckpointError("checkpoint config is invalid") from exc
    state = payload.get("state")
    if not isinstance(state, dict):
        raise CorruptedCheckpointError("checkpoint state is not a mapping")
    return model_name, config, state


def _strip_legacy_hyperparameters(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    payload = dict(node)
    hyperparameters = payload.pop("hyperparameters", None)
    if isinstance(hyperparameters, dict) and "featurizers" in hyperparameters:
        payload["featurizers"] = [
            _strip_legacy_hyperparameters(child) for child in hyperparameters.get("featurizers", [])
        ]
    if "featurizers" in payload and isinstance(payload["featurizers"], list):
        payload["featurizers"] = [_strip_legacy_hyperparameters(child) for child in payload["featurizers"]]
    return payload


def _legacy_featurizer_slot(registry: str) -> str:
    return "cell_line_featurizer" if registry == "cell_line" else "drug_featurizer"


def _collect_legacy_featurizer_overrides(
    node: Any,
    registry: str,
    overrides: dict[str, Any],
) -> None:
    from drevalpy.components.featurizer_label import qualified_featurizer_selector

    if not isinstance(node, dict):
        return
    hyperparameters = node.get("hyperparameters") or {}
    if not isinstance(hyperparameters, dict):
        return
    name = str(node.get("name", ""))
    if name == "concatFeaturizers":
        for child in hyperparameters.get("featurizers", []):
            _collect_legacy_featurizer_overrides(child, registry, overrides)
        return
    view = node.get("view")
    selector = qualified_featurizer_selector(name, view if isinstance(view, str) else None)
    slot = _legacy_featurizer_slot(registry)
    for key, value in hyperparameters.items():
        if key in {"featurizers", "view", "views"}:
            continue
        overrides[f"{slot}.{selector}.{key}"] = value


def _collect_legacy_predictor_overrides(
    legacy: dict[str, Any],
    template: ModelConfig,
    overrides: dict[str, Any],
) -> None:
    predictor = legacy.get("predictor")
    if not isinstance(predictor, dict):
        return
    name = str(predictor.get("name", template.predictor.name))
    for key, value in (predictor.get("hyperparameters") or {}).items():
        overrides[f"predictor.{name}.{key}"] = value


def _touch_template_featurizer_leaves(template: ModelConfig) -> None:
    from drevalpy.components.featurizer_tree import iter_featurizer_leaves

    if template.cell_line_featurizer is not None:
        list(iter_featurizer_leaves(template.cell_line_featurizer, "cell_line"))
    if template.drug_featurizer is not None:
        list(iter_featurizer_leaves(template.drug_featurizer, "drug"))


def _legacy_concrete_overrides(legacy: dict[str, Any], template: ModelConfig) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    _collect_legacy_featurizer_overrides(legacy.get("cell_line_featurizer"), "cell_line", overrides)
    _collect_legacy_featurizer_overrides(legacy.get("drug_featurizer"), "drug", overrides)
    _collect_legacy_predictor_overrides(legacy, template, overrides)
    _touch_template_featurizer_leaves(template)
    return overrides


def load_model(path: str | Path) -> DRPModel:
    """Reconstruct a fitted ``DRPModel`` from a checkpoint archive path.

    Reads the stored model name and ``ModelConfig``, builds the matching class
    via ``construct_model``, then restores fitted state. Use this when you do not
    already have a class handle for ``ModelClass.load(path)``.

    Custom featurizers and predictors must already be registered (same as for
    training). Load only artifacts created with ``save_model`` in the same
    drevalpy version family.

    :param path: Archive file path; ``.zip`` is appended when missing.
    :returns: Fitted ``DRPModel`` instance.
    """
    from drevalpy.models._construct_model_api import construct_model

    model_name, config, _state = load_model_payload(path)
    return construct_model(model_name, config).load(path)
