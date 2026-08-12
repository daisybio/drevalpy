"""Single source of truth for the MuData accessors backing each omics view.

The public omics names used in configs, recipes and predictor code are stable. The
modality keys actually present in the published ``.h5mu`` files are not necessarily
identical to them, so every omics access in the package resolves through
:func:`backing_modality` rather than hard-coding a modality string.

Resolution prefers the name as written and falls back to :data:`OMICS_ACCESSORS`,
so a single code path reads both dataset generations: files storing the physical
name and files already carrying the public one.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from types import MappingProxyType
from typing import Final

#: Maps the stable public omics name to the modality key stored in the MuData.
#:
#: ``copy_number_variation_gistic`` is the name the library, the zoo presets and user
#: recipes all use, but the published datasets store the modality as
#: ``copy_number_variation``. Adding the ``_gistic`` suffix to the datasets is the
#: planned long-term fix. Because :func:`backing_modality` prefers whichever name the
#: file actually has, datasets can be renamed one at a time without touching any code;
#: this entry can then be reduced to an identity mapping, or dropped, at leisure.
OMICS_ACCESSORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "gene_expression": "gene_expression",
        "methylation": "methylation",
        "mutations": "mutations",
        "proteomics": "proteomics",
        "copy_number_variation_gistic": "copy_number_variation",
    }
)


#: Inverse of :data:`OMICS_ACCESSORS`, derived rather than maintained by hand so the
#: dict above stays the only thing to edit. The mapping is injective, so this is exact.
_PUBLIC_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {accessor: public for public, accessor in OMICS_ACCESSORS.items()}
)


def resolve_omics_accessor(name: str) -> str:
    """Translate a public omics name into the modality key to read from the MuData.

    Names that are not registered omics views are returned unchanged, so custom
    matrices and non-omics views keep working.

    Args:
        name: Public omics name, e.g. ``"copy_number_variation_gistic"``.

    Returns:
        The modality key to look up in the MuData.
    """
    return OMICS_ACCESSORS.get(name, name)


def public_omics_name(accessor: str) -> str:
    """Translate a stored modality key back into the name users write.

    The inverse of :func:`resolve_omics_accessor`, for messages and listings: a
    dataset physically holding ``copy_number_variation`` is reported as offering
    ``copy_number_variation_gistic``, which is the name a config may ask for.

    Args:
        accessor: Modality key as stored in the MuData.

    Returns:
        The public omics name, or *accessor* unchanged if it is not an omics view.
    """
    return _PUBLIC_NAMES.get(accessor, accessor)


def backing_modality(name: str, available: Container[str]) -> str | None:
    """Pick the modality key in *available* that backs the public omics *name*.

    The name as written wins, and :data:`OMICS_ACCESSORS` is only a fallback. That
    ordering is what lets one code path serve both dataset generations: files that
    store ``copy_number_variation`` are reached through the map, and files that
    already carry the suffixed name are reached directly. So the eventual dataset
    rename needs no code change at all, not even to the map.

    Args:
        name: Public omics name, e.g. ``"copy_number_variation_gistic"``.
        available: Modality keys present in the MuData.

    Returns:
        The key to read, or ``None`` when nothing in *available* backs *name*.
    """
    if name in available:
        return name
    accessor = OMICS_ACCESSORS.get(name)
    if accessor is not None and accessor in available:
        return accessor
    return None
