"""Model naming and result path helpers for experiments."""

from __future__ import annotations

import numpy as np
from upath import UPath as Path

from ..models._model_lookup import (
    is_multi_drug_model_name,
    is_single_drug_model_name,
)
from ..models.drp_model import DRPModel


def make_model_list(models: list[type[DRPModel]], drug_ids: np.ndarray) -> dict[str, str]:
    """Build model run keys (including per-drug keys for single-drug models).

    :param models: Model classes to include in the run.
    :param drug_ids: Array of drug identifiers to enumerate single-drug keys.

    :returns: Mapping from run key to base model name.
    """
    model_list: dict[str, str] = {}
    unique_drugs = np.unique(drug_ids)
    for model in models:
        if model.is_single_drug():
            for drug in unique_drugs:
                model_list[f"{model.get_model_name()}.{drug}"] = model.get_model_name()
        else:
            model_list[model.get_model_name()] = model.get_model_name()
    return model_list


def get_model_name_and_drug_id(model_name: str) -> tuple[str, str | None]:
    """Parse a run key into model name and optional drug id.

    :param model_name: Run key, optionally suffixed with ``.<drug_id>``.

    :returns: Base model name and drug id, or ``None`` for multi-drug models.

    :raises AssertionError: If the base model name is not recognized.
    """
    if is_multi_drug_model_name(model_name):
        return model_name, None
    name_split = model_name.split(".")
    parsed_name = name_split[0]
    if not is_single_drug_model_name(parsed_name):
        raise AssertionError(
            f"Model {parsed_name} not found in the built-in or external zoo. "
            "Register a zoo preset or pass a known model name."
        )
    return parsed_name, name_split[1]


def generate_data_saving_path(
    model_name: str,
    drug_id: str | None,
    result_path: str | Path,
    suffix: str,
) -> Path:
    """Return output directory for predictions, hpams, and similar artifacts.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :param suffix: Subdirectory label (for example ``predictions``).

    :returns: Created output directory path.
    """
    root = Path(result_path)
    if is_single_drug_model_name(model_name):
        model_path = root / model_name / "drugs" / str(drug_id) / suffix
    else:
        model_path = root / model_name / suffix
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def generate_final_model_checkpoint_path(
    model_name: str,
    drug_id: str | None,
    result_path: str | Path,
) -> Path:
    """Return archive path stem for a final production model checkpoint.

    Creates the model (and optional drug) parent directory only. ``save_model``
    appends ``.zip`` when missing, so this must not create a directory at the
    returned path itself.

    :param model_name: Base model name.
    :param drug_id: Drug identifier for single-drug models.
    :param result_path: Experiment result root directory.
    :returns: Checkpoint path stem (for example ``.../ElasticNet/final_model``).
    """
    root = Path(result_path)
    if is_single_drug_model_name(model_name):
        parent = root / model_name / "drugs" / str(drug_id)
    else:
        parent = root / model_name
    parent.mkdir(parents=True, exist_ok=True)
    return parent / "final_model"
