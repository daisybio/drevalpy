"""Internal model-name resolution that does not use deprecated factory dicts."""

from __future__ import annotations

from drevalpy.models._construct_model_api import construct_model
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.zoo import get_zoo_config, list_zoo_names
from drevalpy.types.enums.model_scope import ModelScope


def get_model_class(name: str) -> type[DRPModel]:
    """Return the ``DRPModel`` facade class for a zoo preset name.

    :param name: Built-in or external zoo preset name.
    :returns: Generated ``DRPModel`` subclass for the preset.
    """
    return construct_model(name)


def known_model_names(*, include_external: bool = True) -> list[str]:
    """Return sorted zoo model names available for CLI/experiment resolution.

    :param include_external: Include externally registered zoo entries.
    :returns: Sorted list of resolvable model names.
    """
    return list_zoo_names(include_external=include_external)


def is_single_drug_model_name(name: str) -> bool:
    """Return whether *name* is a single-drug zoo preset.

    :param name: Built-in or external zoo preset name.
    :returns: ``True`` when the preset scope is single-drug; ``False`` if unknown.
    """
    try:
        return get_zoo_config(name).scope == ModelScope.SINGLE_DRUG
    except KeyError:
        return False


def is_multi_drug_model_name(name: str) -> bool:
    """Return whether *name* is a multi-drug zoo preset.

    :param name: Built-in or external zoo preset name.
    :returns: ``True`` when the preset scope is multi-drug; ``False`` if unknown.
    """
    try:
        return get_zoo_config(name).scope == ModelScope.MULTI_DRUG
    except KeyError:
        return False


def single_drug_model_names(*, include_external: bool = True) -> list[str]:
    """Return sorted single-drug zoo names.

    :param include_external: Include externally registered zoo entries.
    :returns: Sorted list of single-drug preset names.
    """
    return list_zoo_names(include_external=include_external, scope=ModelScope.SINGLE_DRUG)


def multi_drug_model_names(*, include_external: bool = True) -> list[str]:
    """Return sorted multi-drug zoo names.

    :param include_external: Include externally registered zoo entries.
    :returns: Sorted list of multi-drug preset names.
    """
    return list_zoo_names(include_external=include_external, scope=ModelScope.MULTI_DRUG)
