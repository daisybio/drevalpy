"""Validation of the literature reference a component is registered with."""

from __future__ import annotations

from drevalpy.registry.components._metadata import LiteratureReference


def _is_valid_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def validate_literature_reference(reference: LiteratureReference) -> list[str]:
    """Return invalid-field names for a literature reference, or an empty list.

    :param reference: Literature reference attached to a registered class.
    :returns: Names of the invalid fields (``repo_url``, ``citation``, ``deviations``).
    """
    invalid: list[str] = []
    if not reference.repo_url or not _is_valid_url(reference.repo_url):
        invalid.append("repo_url")
    if not (reference.citation_text or reference.citation_doi):
        invalid.append("citation")
    if not reference.deviations:
        invalid.append("deviations")
    return invalid
