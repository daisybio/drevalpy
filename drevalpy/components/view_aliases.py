"""Canonical omics view names and user-facing aliases."""

from __future__ import annotations

CANONICAL_OMICS_VIEWS: frozenset[str] = frozenset(
    {
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
        "proteomics",
    }
)

_OMICS_VIEW_ALIASES: dict[str, str] = {
    "expression": "gene_expression",
    "gene_expression": "gene_expression",
    "methylation": "methylation",
    "mutations": "mutations",
    "proteomics": "proteomics",
    "cnv": "copy_number_variation_gistic",
    "copy_number": "copy_number_variation_gistic",
    "copy_number_variation": "copy_number_variation_gistic",
    "copy_number_variation_gistic": "copy_number_variation_gistic",
}

_PREFERRED_ALIAS_BY_VIEW: dict[str, str] = {
    "gene_expression": "expression",
    "methylation": "methylation",
    "mutations": "mutations",
    "proteomics": "proteomics",
    "copy_number_variation_gistic": "cnv",
}


def resolve_omics_view(token: str) -> str:
    """Map a user-facing view token to the canonical storage key."""
    normalized = token.strip().lower()
    if not normalized:
        msg = "Omics view token must be a non-empty string"
        raise ValueError(msg)
    if normalized in CANONICAL_OMICS_VIEWS:
        return normalized
    if normalized in _OMICS_VIEW_ALIASES:
        return _OMICS_VIEW_ALIASES[normalized]
    allowed = ", ".join(sorted(_OMICS_VIEW_ALIASES))
    msg = f"Unknown omics view {token!r}. Allowed aliases: {allowed}"
    raise ValueError(msg)


def format_view_alias(view: str) -> str:
    """Format a canonical view key for bracket labels."""
    return _PREFERRED_ALIAS_BY_VIEW.get(view, view)
