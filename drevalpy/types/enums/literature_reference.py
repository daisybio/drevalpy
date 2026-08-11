"""Structured literature citation metadata for registered components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureReference:
    """Repository, citation, and integration-deviation notes for a literature port."""

    repo_url: str
    citation_text: str = ""
    citation_doi: str = ""
    deviations: str = ""

    def __post_init__(self) -> None:
        """Normalize surrounding whitespace in every field."""
        object.__setattr__(self, "repo_url", self.repo_url.strip())
        object.__setattr__(self, "citation_text", self.citation_text.strip())
        object.__setattr__(self, "citation_doi", self.citation_doi.strip())
        object.__setattr__(self, "deviations", self.deviations.strip())
