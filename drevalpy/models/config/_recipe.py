"""The recipe language: one grammar for reading it, one function for writing it.

A recipe is how a model is named in configs, on the CLI, and in ``ModelConfig.model_id``.
The grammar covers all three layers, so the delimiters are defined in a single place::

    model      := slot ":" slot ":" predictor | slot ":" predictor | predictor
    slot       := atom ("+" atom)*
    atom       := NAME ("[" VIEW "]")?

Only *shape* is checked here. Whether a name exists, whether a featurizer requires a view,
and whether a view alias resolves are semantic questions answered by the registry and by
``drevalpy.components.view_aliases``.
"""

from __future__ import annotations

import pyparsing as pp

_NAME = pp.Regex(r"[^\[\]+:\s]+")
"""Featurizer or predictor name: anything that is not a delimiter or whitespace."""

_VIEW = pp.Regex(r"[^\[\]:]+")
"""View token. Deliberately allows ``+`` so ``raw[a+b]`` stays a single atom and the error
names the unknown view instead of a truncated featurizer name."""

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


def parse_model_recipe(spec: str) -> tuple[str | None, str | None, str]:
    """Split a model recipe into its featurizer slots and predictor name.

    The slots are returned as unparsed recipe strings, since each is normalized separately
    against its own registry. Splitting happens through the grammar rather than on ``:``,
    so a colon inside a view cannot be mistaken for a slot separator.

    :param spec: ``predictor``, ``cell:predictor``, or ``cell:drug:predictor``.
    :returns: ``(cell_line_recipe, drug_recipe, predictor_name)``; slots are ``None`` when absent.
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
    return (
        cell_line.strip() if cell_line is not None else None,
        drug.strip() if drug is not None else None,
        parsed["predictor"],
    )


def format_model_recipe(cell_line: str | None, drug: str | None, predictor: str) -> str:
    """Join component names back into a model recipe.

    The inverse of ``parse_model_recipe``, and the only place the slot separator is written
    out. A recipe names its slots left to right, so a drug slot without a cell-line slot has
    nowhere to go.

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
