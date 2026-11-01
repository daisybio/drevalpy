"""Registration of predictor and featurizer classes, and the catalog facts listed for them.

The predictor and featurizer ``register`` decorators share :func:`component_decorator`: it
checks the description, tags and reference when the decorator is created, then rejects a class
with unimplemented abstract members (a component that forgets ``_fit`` would otherwise only fail
when an experiment instantiates it), runs the kind's own checks and registers the class. The
description and tags live once, in the registry entry's metadata; the class gets only its
``reference``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from typing import Any, TypeVar

from drevalpy.components.predictors.inputs import predictor_inputs
from drevalpy.registry._registry import Registry

C = TypeVar("C", bound=type[Any])


@dataclass(frozen=True)
class LiteratureReference:
    """Repository, citation, and integration-deviation notes for a literature port.

    Every field is stripped of surrounding whitespace. ``repo_url`` must be an ``http(s)`` URL,
    and a citation (``citation_text`` or ``citation_doi``) and ``deviations`` must be given.
    """

    repo_url: str
    citation_text: str = ""
    citation_doi: str = ""
    deviations: str = ""

    def __post_init__(self) -> None:
        """Strip every field and reject a reference with missing or malformed fields.

        :raises ValueError: If ``repo_url``, the citation or ``deviations`` is missing or invalid.
        """
        for field in fields(self):
            object.__setattr__(self, field.name, getattr(self, field.name).strip())
        invalid = [
            name
            for name, valid in (
                ("repo_url", self.repo_url.startswith(("http://", "https://"))),
                ("citation", bool(self.citation_text or self.citation_doi)),
                ("deviations", bool(self.deviations)),
            )
            if not valid
        ]
        if invalid:
            msg = f"reference has invalid fields: {invalid}"
            raise ValueError(msg)

    @property
    def citation(self) -> str:
        """The DOI as a resolvable URL when given, else the citation text."""
        cite = self.citation_doi or self.citation_text
        return f"https://doi.org/{cite}" if cite.startswith("10.") else cite


def _clean_tags(tags: Iterable[str] | None) -> frozenset[str]:
    """Return the non-blank *tags*, stripped.

    :param tags: Discovery tags given at registration, or ``None``.
    :returns: Stripped tags without blanks.
    :raises TypeError: If *tags* is a bare string or holds a non-string.
    """
    if isinstance(tags, str):
        msg = "tags must be an iterable of strings, not a bare string"
        raise TypeError(msg)
    cleaned: set[str] = set()
    for tag in tags or ():
        if not isinstance(tag, str):
            msg = f"tags must contain strings, got {type(tag).__name__}"
            raise TypeError(msg)
        if tag.strip():
            cleaned.add(tag.strip())
    return frozenset(cleaned)


def component_decorator(
    registry: Registry[type[Any]],
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None,
    reference: LiteratureReference | None,
    validate: Callable[[str, type[Any]], None],
) -> Callable[[C], C]:
    """Return a class decorator that checks a component and registers it under *name*.

    A duplicate *name* makes :meth:`Registry.register` raise ``ValueError`` when the decorator runs.

    :param registry: Registry of the component kind.
    :param name: Registry name of the component.
    :param description: Short human-readable summary, stored in the entry's metadata.
    :param tags: Optional discovery tags, stored in the entry's metadata.
    :param reference: Optional literature citation, set as the class's ``reference``.
    :param validate: Invariants of the component kind, called as ``validate(name, cls)``.
    :returns: Decorator that registers and returns the decorated class.
    :raises ValueError: If *description* is blank, or (when decorating) the class still has
        unimplemented abstract members.
    :raises TypeError: If *reference* is not a :class:`LiteratureReference`; bad *tags* raise
        ``TypeError`` too.
    """
    stripped = str(description).strip()
    if not stripped:
        msg = "description must be a non-empty string"
        raise ValueError(msg)
    if reference is not None and not isinstance(reference, LiteratureReference):
        msg = f"reference must be LiteratureReference, got {type(reference).__name__}"
        raise TypeError(msg)
    metadata = {"description": stripped, "tags": _clean_tags(tags)}

    def decorate(cls: C) -> C:
        missing = sorted(getattr(cls, "__abstractmethods__", frozenset()))
        if missing:
            msg = (
                f"{registry.kind} '{name}' ({cls.__name__}) does not implement {', '.join(missing)}. "
                "Implement the missing member(s), or register a concrete subclass instead of an abstract base."
            )
            raise ValueError(msg)
        validate(name, cls)
        registry.register(name, cls, metadata=metadata)
        cls.reference = reference
        return cls

    return decorate


def _component_metadata(registry: Registry[type[Any]], registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return the catalog facts every component shares.

    :param registry: Registry holding *name*, whose metadata gives the description and tags.
    :param registry_name: Catalog name of the registry, reported as ``registry``.
    :param name: Name the class is registered under.
    :param cls: Registered component class.
    :returns: Catalog metadata dict.
    """
    entry = registry.metadata(name)
    reference = getattr(cls, "reference", None)
    if not isinstance(reference, LiteratureReference):
        reference = None
    return {
        "registry": registry_name,
        "name": name,
        "class_name": cls.__name__,
        "description": entry.get("description", ""),
        "tags": entry.get("tags", frozenset()),
        "repo_url": reference.repo_url if reference else "",
        "citation": reference.citation if reference else "",
        "citation_doi": reference.citation_doi if reference else "",
        "citation_text": reference.citation_text if reference else "",
        "deviations": reference.deviations if reference else "",
    }


def _output_format(cls: type[Any]) -> str:
    try:
        blocks = cls.output_blocks({})
    except Exception:  # noqa: BLE001 - a config-dependent featurizer has no default-config blocks
        return ""
    return ", ".join(dict.fromkeys(block.format.value for block in blocks))


def featurizer_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return the catalog facts of a registered featurizer: the shared ones, its sides and output format.

    A featurizer that declares its blocks through ``output_blocks`` reports the formats of its
    default-config blocks, joined; one that emits none reports ``""``.

    :param registry_name: Catalog name of the registry, reported as ``registry``.
    :param name: Name the class is registered under in the featurizer registry.
    :param cls: Registered featurizer class.
    :returns: Catalog metadata dict.
    """
    from drevalpy.registry.featurizer import featurizers

    meta = _component_metadata(featurizers, registry_name, name, cls)
    meta["sides"] = frozenset(getattr(cls, "sides", ()))
    meta["output_format"] = _output_format(cls)
    return meta


def predictor_component_metadata(registry_name: str, name: str, cls: type[Any]) -> dict[str, Any]:
    """Return the catalog facts of a registered predictor: the shared ones and its input interface.

    The input interface is ``"block"`` when either side reads named blocks, else ``"matrix"``
    when either side reads a matrix, else ``"feature_free"``.

    :param registry_name: Catalog name of the registry, reported as ``registry``.
    :param name: Name the class is registered under in the predictor registry.
    :param cls: Registered predictor class.
    :returns: Catalog metadata dict.
    """
    from drevalpy.registry.predictor import predictor_registry

    meta = _component_metadata(predictor_registry, registry_name, name, cls)
    kinds = {predictor_inputs(cls, "sample").kind, predictor_inputs(cls, "drug").kind}
    meta["input_interface"] = "block" if "blocks" in kinds else "matrix" if "matrix" in kinds else "feature_free"
    return meta
