"""Read every accepted spelling of a model config into canonical fields, and write a model back as a recipe.

A recipe is how a model is named in configs, on the CLI, and in ``ModelConfig.model_id``.
One grammar covers all three layers, so the delimiters are defined in a single place::

    model      := slot ":" slot ":" predictor | slot ":" predictor | predictor
    slot       := atom ("+" atom)*
    atom       := NAME ("[" VIEW "]")?

``+`` stands for a ``concatFeaturizers`` node and brackets for a ``view`` field. Expanding a
recipe is a transcription, not an interpretation: it produces the same mapping a YAML file would
have spelled out, so a recipe and the YAML for the same model are the same input from there on.
Only *shape* is checked here; whether a name exists or a view resolves is answered downstream
for every spelling alike.

A featurizer slot is either a recipe string or the canonical mapping;
:func:`normalize_featurizer` is the one place that reads them::

    "scaledGeneExpression"                     name=scaledGeneExpression
    "view[gene_expression]"                     name=raw, view=gene_expression
    "view[gene_expression]+view[mutations]"      name=concatFeaturizers, two children
    {"name": "pca", "view": "methylation",     canonical: name, view, featurizers (concat
     "options": {...}, "defaults": {...}}      children), options, hyperparameter_space, defaults

A predictor slot is a bare name or the canonical mapping ``{name, hyperparameter_space,
defaults}`` (:func:`normalize_predictor`). ``options`` are fixed constructor values and may not
name a tunable hyperparameter; ``defaults`` moves the ``default`` of entries of the search space
(the declared ``hyperparameter_space``, else the component's own) and is folded into
``hyperparameter_space`` here, so a config records the complete space it tunes over. The list
form and the one-key mapping form (``{"fingerprints": {"n_bits": 512}}``) are rejected with the
canonical mapping to write instead.

:func:`format_model_recipe` writes the grammar back; :attr:`ModelConfig.model_id` uses it, so a
model id reads back through ``from_recipe`` into the same components.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

import pyparsing as pp

from drevalpy.components.featurizers._featurizer_label import CONCAT
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.registry.predictor import get as get_predictor

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig

_NAME = pp.Regex(r"[^\[\]+:\s]+")
"""Featurizer or predictor name: anything that is not a delimiter or whitespace."""

_VIEW = pp.Regex(r"[^\[\]:]+")
"""View token. Deliberately allows ``+`` so ``view[a+b]`` stays a single atom and the error
names the unusable view instead of a truncated featurizer name."""

_SLOT_SEP = ":"
_ATOM_SEP = "+"

_ATOM = pp.Group(_NAME("name") + pp.Optional(pp.Suppress("[") + _VIEW("view") + pp.Suppress("]")))
_FEATURIZERS = pp.DelimitedList(_ATOM, delim=_ATOM_SEP)
_SLOT = pp.original_text_for(_FEATURIZERS)
_COLON = pp.Suppress(_SLOT_SEP)

_FEATURIZER_RECIPE = _FEATURIZERS + pp.StringEnd()
_MODEL_RECIPE = (
    _SLOT("sample") + _COLON + _SLOT("drug") + _COLON + _NAME("predictor")
    | _SLOT("sample") + _COLON + _NAME("predictor")
    | _NAME("predictor")
) + pp.StringEnd()

_FEATURIZER_SYNTAX = "atoms must be non-empty and shaped 'name' or 'name[view]', joined by '+'"
_MODEL_SYNTAX = "expected 'predictor', 'sampleFeaturizer:predictor', or 'sampleFeaturizer:drugFeaturizer:predictor'"


def _require_text(text: str, what: str) -> str:
    """Return *text* stripped, rejecting a blank one.

    :param text: Recipe text.
    :param what: What the text is, for the message.
    :returns: The stripped text.
    :raises ValueError: If *text* is blank.
    """
    trimmed = text.strip()
    if not trimmed:
        raise ValueError(f"{what} must be a non-empty string")
    return trimmed


def parse_featurizer_atoms(token: str) -> list[tuple[str, str | None]]:
    """Parse a featurizer recipe into its atoms.

    :param token: Recipe string such as ``"view[gene_expression]+scaledGeneExpression"``.
    :returns: One ``(name, view)`` pair per atom, with *view* ``None`` when unbracketed.
    :raises ValueError: If *token* is not a well-formed featurizer recipe.
    """
    try:
        parsed = _FEATURIZER_RECIPE.parse_string(token, parse_all=True)
    except pp.ParseBaseException as exc:
        raise ValueError(f"Malformed featurizer recipe {token!r}: {_FEATURIZER_SYNTAX}") from exc
    return [(atom["name"], atom.get("view")) for atom in parsed]


def expand_featurizer_recipe(token: str) -> dict[str, Any]:
    """Expand a featurizer recipe into the field mapping a YAML file would have spelled out.

    A single atom expands to one mapping; ``+``-joined atoms expand to a concat node over them.
    A blank or malformed *token* raises ``ValueError``.

    :param token: Recipe string such as ``"view[gene_expression]+scaledGeneExpression"``.
    :returns: Featurizer or concat-featurizer field mapping.
    """
    atoms = parse_featurizer_atoms(_require_text(token, "Featurizer token"))
    payloads = [{"name": name} if view is None else {"name": name, "view": view} for name, view in atoms]
    if len(payloads) == 1:
        return payloads[0]
    return {"name": CONCAT, "featurizers": payloads}


def parse_model_recipe(spec: str) -> dict[str, Any]:
    """Read a model recipe into the plain field mapping a model config is built from.

    This is to a recipe string what ``yaml.safe_load`` is to a YAML file: source syntax in,
    plain mapping out, no registry involved. Splitting happens through the grammar rather than
    on ``:``, so a colon inside a view cannot be mistaken for a slot separator.

    :param spec: ``predictor``, ``sample:predictor``, or ``sample:drug:predictor``.
    :returns: ``sample_featurizer``, ``drug_featurizer`` and ``predictor`` entries; the two
        featurizer slots are ``None`` when the recipe omits them.
    :raises ValueError: If *spec* is blank or not a well-formed model recipe.
    """
    trimmed = _require_text(spec, "model recipe")
    try:
        parsed = _MODEL_RECIPE.parse_string(trimmed, parse_all=True)
    except pp.ParseBaseException as exc:
        raise ValueError(f"Malformed model recipe {spec!r}: {_MODEL_SYNTAX}") from exc
    sample = parsed.get("sample")
    drug = parsed.get("drug")
    return {
        "sample_featurizer": expand_featurizer_recipe(sample) if sample is not None else None,
        "drug_featurizer": expand_featurizer_recipe(drug) if drug is not None else None,
        "predictor": parsed["predictor"],
    }


def format_featurizer_recipe(config: FeaturizerConfig) -> str:
    """Write a featurizer tree as the slot of a recipe, the inverse of :func:`expand_featurizer_recipe`.

    :param config: Featurizer config, possibly a concat node.
    :returns: ``name``, ``name[view]``, or such atoms joined by ``+`` for a concat node.
    """
    return _ATOM_SEP.join(leaf.selector for leaf in config.leaves())


def format_model_recipe(sample: str | None, drug: str | None, predictor: str) -> str:
    """Join recipe slots into a model recipe.

    Writes the grammar that :func:`parse_model_recipe` reads, and the only place the slot
    separator is written out. A recipe names its slots left to right, so a drug slot without
    a sample slot has nowhere to go.

    :param sample: Sample featurizer slot, or ``None`` for feature-free predictors.
    :param drug: Drug featurizer slot, or ``None`` when omitted.
    :param predictor: Predictor name.
    :returns: Model recipe of one to three colon-separated parts.
    :raises ValueError: If *predictor* is empty, or *drug* is set without *sample*.
    """
    if not predictor:
        raise ValueError("predictor is required")
    if sample is None and drug is None:
        return predictor
    if sample is None:
        raise ValueError("sample is required when drug is set")
    parts = [sample, predictor] if drug is None else [sample, drug, predictor]
    return _SLOT_SEP.join(parts)


def _space_with_defaults(cls: type[Any], declared: Any, defaults: Any, what: str) -> dict[str, Any]:
    """Return the search space of a component with the ``default`` of some entries moved.

    :param cls: Component class exposing ``get_hyperparameter_space``.
    :param declared: The config's ``hyperparameter_space``, or ``None`` for the class's own.
    :param defaults: Mapping of local hyperparameter name to its new default.
    :param what: The component, for messages (``"featurizer 'fingerprints'"``).
    :returns: The complete space with the defaults applied.
    :raises ValueError: If *defaults* is not a mapping or names a key that is not tunable.
    """
    if not isinstance(defaults, Mapping):
        raise ValueError(f"defaults of {what} must be a mapping of hyperparameter name to value")
    base = cls.get_hyperparameter_space() if declared is None else declared
    space = {key: dict(spec) if isinstance(spec, Mapping) else spec for key, spec in base.items()}
    tunable = sorted(key for key, spec in space.items() if isinstance(spec, dict))
    for key, value in defaults.items():
        if key not in tunable:
            raise ValueError(
                f"defaults of {what}: {key!r} is not a tunable hyperparameter "
                f"(tunable: {', '.join(tunable) or 'none'}); put fixed constructor values under 'options'"
            )
        space[key] = {**space[key], "default": value}
    return space


def _reject_tunable_options(cls: type[Any], declared: Any, options: Any, what: str) -> None:
    """Reject ``options`` that name a tunable hyperparameter, whose value tuning would override.

    :param cls: Featurizer class exposing ``get_hyperparameter_space``.
    :param declared: The config's ``hyperparameter_space``, or ``None`` for the class's own.
    :param options: The config's ``options``.
    :param what: The component, for messages.
    :raises ValueError: If an option key is an entry of the search space.
    """
    space = cls.get_hyperparameter_space() if declared is None else declared
    clashes = sorted(key for key in options if isinstance(space.get(key), Mapping))
    if clashes:
        raise ValueError(
            f"options of {what} name tunable hyperparameters ({', '.join(clashes)}); "
            "set their value under 'defaults' instead"
        )


def _canonical_hint(name_fields: dict[str, Any], body: Any, loose_field: str) -> dict[str, Any]:
    """Spell a one-key mapping ``{token: body}`` as the canonical mapping, for an error message.

    :param name_fields: ``name`` (and ``view``) read from the key.
    :param body: The mapping's value.
    :param loose_field: Where the body's loose values go (``"options"`` or ``"defaults"``).
    :returns: The canonical mapping to suggest.
    """
    hint = dict(name_fields)
    loose = dict(body) if isinstance(body, Mapping) else {}
    for field in ("featurizers", "options", "hyperparameter_space"):
        if field in loose:
            hint[field] = loose.pop(field)
    if loose:
        hint[loose_field] = {**loose, **hint.get(loose_field, {})}
    return hint


def _featurizer_hint(raw: Any) -> Any:
    """Spell a featurizer config in an accepted form, for an error message; *raw* if it already is one.

    :param raw: A featurizer config in any spelling.
    :returns: The recipe string or canonical mapping to suggest.
    """
    if isinstance(raw, list):
        if raw and all(isinstance(item, str) for item in raw):
            return _ATOM_SEP.join(raw)
        return {"name": CONCAT, "featurizers": [_featurizer_hint(item) for item in raw]}
    if not isinstance(raw, dict) or "name" in raw or len(raw) != 1:
        return raw
    token, body = next(iter(raw.items()))
    try:
        name_fields = expand_featurizer_recipe(str(token))
    except ValueError:
        name_fields = {"name": str(token)}
    hint = _canonical_hint(name_fields, body, "options")
    if isinstance(hint.get("featurizers"), list):
        hint["featurizers"] = [_featurizer_hint(item) for item in hint["featurizers"]]
    return hint


def _reject_removed_featurizer_spelling(raw: list[Any] | dict[str, Any]) -> NoReturn:
    """Reject a list or a featurizer mapping without ``name``, showing the canonical form to write.

    :param raw: The rejected featurizer config.
    :raises ValueError: Always.
    """
    if isinstance(raw, list):
        raise ValueError(f"A featurizer list is not accepted; write {_featurizer_hint(raw)!r}")
    if len(raw) != 1:
        raise ValueError(f"A featurizer mapping needs a 'name' key, got keys {sorted(map(str, raw))}")
    token, hint = next(iter(raw)), _featurizer_hint(raw)
    raise ValueError(
        f"The one-key featurizer mapping {{{token!r}: ...}} is not accepted; write {hint!r}, "
        "with tunable hyperparameters under 'defaults' instead of 'options'"
    )


def _resolve_side(fields: dict[str, Any], side: str | None) -> str:
    """Return the side of a featurizer mapping: its own ``side``, else *side*.

    :param fields: Featurizer field mapping.
    :param side: Side of the slot or concat parent holding it, or ``None``.
    :returns: The side's value.
    :raises ValueError: If neither gives a side, or the two disagree.
    """
    own = fields.get("side")
    if own is None and side is None:
        raise ValueError(
            f"featurizer {fields['name']!r} needs a side: put it in a model config's sample_featurizer or "
            "drug_featurizer slot, or set 'side'"
        )
    if own is not None and side is not None and str(own) != str(side):
        raise ValueError(f"featurizer {fields['name']!r} has side {str(own)!r} but is used on the {side} side")
    return str(side if own is None else own)


def _featurizer_fields(raw: Any) -> dict[str, Any]:
    """Return a featurizer config's fields as a fresh mapping, expanding a recipe string.

    :param raw: Recipe string or mapping with ``name``.
    :returns: The fields, not yet normalized.
    :raises TypeError: If *raw* is not a string, list or mapping.
    """
    if isinstance(raw, str):
        return expand_featurizer_recipe(raw)
    if isinstance(raw, list) or (isinstance(raw, dict) and "name" not in raw):
        _reject_removed_featurizer_spelling(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"Featurizer config must be a recipe string or a mapping with 'name', got {type(raw)!r}")
    return dict(raw)


def normalize_featurizer(raw: Any, side: str | None = None) -> dict[str, Any]:
    """Return the canonical ``FeaturizerConfig`` fields of a recipe string or canonical mapping.

    ``defaults`` is folded into ``hyperparameter_space``; children of a concat node are
    normalized the same way and inherit its side. A *raw* that is not a string, list or mapping
    raises ``TypeError``.

    :param raw: Recipe string or mapping with ``name``.
    :param side: Side of the slot holding the featurizer; ``None`` requires *raw* to set ``side``.
    :returns: Canonical field mapping, ``side`` included.
    :raises ValueError: If *raw* is a list, a mapping without ``name``, has no side or another side
        than *side*, or carries ``defaults`` or ``options`` that do not fit the featurizer's search space.
    """
    fields = _featurizer_fields(raw)
    fields["side"] = side = _resolve_side(fields, side)
    what = f"{side} featurizer {fields['name']!r}"
    declared = fields.get("hyperparameter_space")
    if fields.get("options"):
        _reject_tunable_options(get_featurizer(fields["name"], side), declared, fields["options"], what)
    if "defaults" in fields:
        fields["hyperparameter_space"] = _space_with_defaults(
            get_featurizer(fields["name"], side), declared, fields.pop("defaults"), what
        )
    children = fields.get("featurizers")
    if children is not None:
        if not isinstance(children, (list, tuple)):
            raise ValueError("featurizers must be a list when set")
        fields["featurizers"] = [
            child if _is_config(child) else normalize_featurizer(child, side) for child in children
        ]
    return fields


def _is_config(value: Any) -> bool:
    """Return whether *value* is an already validated ``FeaturizerConfig``.

    :param value: A featurizer config in any spelling.
    :returns: ``True`` for a ``FeaturizerConfig`` instance.
    """
    from drevalpy.models.config.featurizer import FeaturizerConfig

    return isinstance(value, FeaturizerConfig)


def normalize_predictor(raw: Any) -> dict[str, Any]:
    """Return the canonical ``PredictorConfig`` fields of a bare name or canonical mapping.

    ``defaults`` is folded into ``hyperparameter_space``.

    :param raw: Bare name, or a mapping with ``name``.
    :returns: Canonical field mapping.
    :raises ValueError: If a mapping has no ``name`` (such as the one-key form), or ``defaults``
        names a key that is not tunable.
    :raises TypeError: If *raw* is not a string or mapping.
    """
    if isinstance(raw, str):
        return {"name": raw}
    if not isinstance(raw, dict):
        raise TypeError(f"Predictor config must be a name or a mapping with 'name', got {type(raw)!r}")
    if "name" not in raw:
        hint = ""
        if len(raw) == 1:
            name, body = next(iter(raw.items()))
            canonical = _canonical_hint({"name": str(name)}, body, "defaults")
            hint = f"; the one-key form is not accepted, write {canonical!r}"
        raise ValueError(f"A predictor mapping needs a 'name' key{hint}")
    fields = dict(raw)
    if "defaults" in fields:
        fields["hyperparameter_space"] = _space_with_defaults(
            get_predictor(fields["name"]),
            fields.get("hyperparameter_space"),
            fields.pop("defaults"),
            f"predictor {fields['name']!r}",
        )
    return fields
