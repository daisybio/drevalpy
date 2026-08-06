"""The recipe language: one grammar for reading it, one function for writing it.

A recipe is how a model is named in configs, on the CLI, and in ``ModelConfig.model_id``.
The grammar covers all three layers, so the delimiters are defined in a single place::

    model      := slot ":" slot ":" predictor | slot ":" predictor | predictor
    slot       := atom ("+" atom)*
    atom       := NAME ("[" VIEW "]")?

This module owns the recipe notation end to end: the grammar above, and what its two
constructs *stand for* -- ``+`` a ``concatFeaturizers`` node, brackets a ``view`` field.
Expanding a recipe is a transcription, not an interpretation: it produces the same mapping a
YAML file would have spelled out, so a recipe and the YAML for the same model are the same
input from there on.

Only *shape* is checked here. Whether a name exists, whether a featurizer takes a view at all,
and whether a view resolves to a matrix are semantic questions, answered downstream for every
notation alike -- a bracket buys no extra scrutiny over a ``view`` key, and no less.
"""

from __future__ import annotations

from typing import Any

import pyparsing as pp

CONCAT_FEATURIZER_NAME = "concatFeaturizers"
"""Registry name of the node a ``+`` recipe stands for."""

_NAME = pp.Regex(r"[^\[\]+:\s]+")
"""Featurizer or predictor name: anything that is not a delimiter or whitespace."""

_VIEW = pp.Regex(r"[^\[\]:]+")
"""View token. Deliberately allows ``+`` so ``raw[a+b]`` stays a single atom and the error
names the unusable view instead of a truncated featurizer name."""

_SLOT_SEP = ":"
_ATOM_SEP = "+"

_ATOM = pp.Group(_NAME("name") + pp.Optional(pp.Suppress("[") + _VIEW("view") + pp.Suppress("]")))
_FEATURIZERS = pp.DelimitedList(_ATOM, delim=_ATOM_SEP)
_SLOT = pp.original_text_for(_FEATURIZERS)
_COLON = pp.Suppress(_SLOT_SEP)

_FEATURIZER_RECIPE = _FEATURIZERS + pp.StringEnd()
_MODEL_RECIPE = (
    _SLOT("cell_line") + _COLON + _SLOT("drug") + _COLON + _NAME("predictor")
    | _SLOT("cell_line") + _COLON + _NAME("predictor")
    | _NAME("predictor")
) + pp.StringEnd()

_FEATURIZER_SYNTAX = "atoms must be non-empty and shaped 'name' or 'name[view]', joined by '+'"
_MODEL_SYNTAX = (
    "expected 'predictor', 'cellLineFeaturizer:predictor', " "or 'cellLineFeaturizer:drugFeaturizer:predictor'"
)


def parse_featurizer_atoms(token: str) -> list[tuple[str, str | None]]:
    """Parse a featurizer recipe into its atoms.

    :param token: Recipe string such as ``"raw[expression]+scaledGeneExpression"``.
    :returns: One ``(name, view)`` pair per atom, with *view* ``None`` when unbracketed.
    :raises ValueError: If *token* is not a well-formed featurizer recipe.
    """
    try:
        parsed = _FEATURIZER_RECIPE.parse_string(token, parse_all=True)
    except pp.ParseBaseException as exc:
        msg = f"Malformed featurizer recipe {token!r}: {_FEATURIZER_SYNTAX}"
        raise ValueError(msg) from exc
    return [(atom["name"], atom.get("view")) for atom in parsed]


def _atom_payload(name: str, view: str | None) -> dict[str, Any]:
    """Transcribe one parsed atom into its field mapping.

    A bracketed view becomes a ``view`` field spelled exactly as written, so an atom carries no
    more and no less information than the mapping form of the same featurizer.

    :param name: Featurizer registry name.
    :param view: View written inside brackets, or ``None`` when unbracketed.
    :returns: Field mapping for this atom.
    """
    if view is None:
        return {"name": name}
    return {"name": name, "view": view}


def expand_featurizer_recipe(token: str) -> dict[str, Any]:
    """Expand a featurizer recipe into the field mapping a YAML file would have spelled out.

    A single atom expands to one mapping; ``+``-joined atoms expand to a concat node over them.
    The result names config fields only, so callers can treat it exactly like a mapping that was
    written out by hand, and it is checked no more strictly than one.

    :param token: Recipe string such as ``"raw[expression]+scaledGeneExpression"``.
    :returns: Featurizer or concat-featurizer field mapping.
    :raises ValueError: If *token* is blank or is not a well-formed recipe.
    """
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    payloads = [_atom_payload(name, view) for name, view in parse_featurizer_atoms(trimmed)]
    if len(payloads) == 1:
        return payloads[0]
    return {"name": CONCAT_FEATURIZER_NAME, "featurizers": payloads}


def parse_model_recipe(spec: str) -> dict[str, Any]:
    """Read a model recipe into the plain field mapping a model config is built from.

    This is to a recipe string what ``yaml.safe_load`` is to a YAML file: source syntax in,
    plain mapping out, no registry involved. Each featurizer slot is expanded into the mapping it
    stands for, so the result is indistinguishable from the same model written as YAML. Splitting
    happens through the grammar rather than on ``:``, so a colon inside a view cannot be
    mistaken for a slot separator.

    :param spec: ``predictor``, ``cell:predictor``, or ``cell:drug:predictor``.
    :returns: ``cell_line_featurizer``, ``drug_featurizer`` and ``predictor`` entries; the two
        featurizer slots are ``None`` when the recipe omits them.
    :raises ValueError: If *spec* is empty or not a well-formed model recipe.
    """
    if not spec or not spec.strip():
        msg = "model recipe must be a non-empty string"
        raise ValueError(msg)
    try:
        parsed = _MODEL_RECIPE.parse_string(spec, parse_all=True)
    except pp.ParseBaseException as exc:
        msg = f"Malformed model recipe {spec!r}: {_MODEL_SYNTAX}"
        raise ValueError(msg) from exc
    cell_line = parsed.get("cell_line")
    drug = parsed.get("drug")
    return {
        "cell_line_featurizer": expand_featurizer_recipe(cell_line) if cell_line is not None else None,
        "drug_featurizer": expand_featurizer_recipe(drug) if drug is not None else None,
        "predictor": parsed["predictor"],
    }


def format_model_recipe(cell_line: str | None, drug: str | None, predictor: str) -> str:
    """Join component names back into a model recipe.

    Writes the grammar that ``parse_model_recipe`` reads, and the only place the slot
    separator is written out. A recipe names its slots left to right, so a drug slot without
    a cell-line slot has nowhere to go.

    :param cell_line: Cell-line featurizer name, or ``None`` for feature-free predictors.
    :param drug: Drug featurizer name, or ``None`` when omitted.
    :param predictor: Predictor name.
    :returns: Model recipe of one to three colon-separated parts.
    :raises ValueError: If *predictor* is empty, or *drug* is set without *cell_line*.
    """
    if not predictor:
        msg = "predictor is required"
        raise ValueError(msg)
    if cell_line is None and drug is None:
        return predictor
    if cell_line is None:
        msg = "cell_line is required when drug is set"
        raise ValueError(msg)
    parts = [cell_line, predictor] if drug is None else [cell_line, drug, predictor]
    return _SLOT_SEP.join(parts)
