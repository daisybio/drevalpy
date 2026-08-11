"""Internal model-name resolution that does not use deprecated factory dicts."""

from __future__ import annotations

from drevalpy.models.zoo import list_zoo_names
from drevalpy.types.enums.model_scope import ModelScope


def known_model_names(*, include_external: bool = True) -> list[str]:
    """Return sorted zoo model names available for CLI/experiment resolution.

    :param include_external: Include externally registered zoo entries.
    :returns: Sorted list of resolvable model names.
    """
    return list_zoo_names(include_external=include_external)


def single_drug_model_names(*, include_external: bool = True) -> list[str]:
    """Return sorted single-drug zoo names.

    :param include_external: Include externally registered zoo entries.
    :returns: Sorted list of single-drug preset names.
    """
    return list_zoo_names(include_external=include_external, scope=ModelScope.SINGLE_DRUG)
