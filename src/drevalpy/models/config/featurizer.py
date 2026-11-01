"""Declarative featurizer configuration schemas."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.components.contracts.hyperparameter_space import validate_hyperparameter_space
from drevalpy.components.featurizers._featurizer_label import CONCAT, requires_explicit_view
from drevalpy.models.config._parse import normalize_featurizer
from drevalpy.models.config.immutable import FrozenMapping, thaw_value
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.types.data.feature_source import Side

if TYPE_CHECKING:
    from drevalpy.components.featurizers.featurizer import Featurizer
    from drevalpy.types.data.batch.feature_block import BlockSpec


class FeaturizerConfig(BaseModel):
    """Immutable template for a featurizer node in a model stack.

    Describes *which* featurizer to build and how to address it, not the built object:
    a ``name`` looked up in the featurizer registry, an optional ``view`` selecting the
    input matrix, ``options`` fixing concrete constructor values, and
    ``hyperparameter_space`` declaring what tuning may vary. ``featurizers`` makes the
    node a tree, holding children for ``concatFeaturizers``. Combine several views by
    nesting one single-``view`` child per view under a concat node, which is what the
    ``view[gene_expression]+view[mutations]`` shorthand expands to. ``side`` says which
    side's registry ``name`` is looked up in; a model config fills it in from the slot holding
    the node, and concat children inherit it.

    Accepts these fields as a mapping, plus ``defaults``, which moves the ``default`` of
    search-space entries and is folded into ``hyperparameter_space``
    (:func:`~drevalpy.models.config._parse.normalize_featurizer`).
    :meth:`parse` also reads a recipe string (``name``, ``name[view]``, ``a+b``) for a given side.
    Validation is front-loaded here so a bad recipe fails at load time instead of mid-run.

    ``tuple`` fields plus ``frozen=True`` make an accidental in-place edit of a shared or
    cached config fail loudly. Note this buys *safety*, not hashability: ``options`` and
    ``hyperparameter_space`` hold arbitrary nested data, so a config carrying either is
    unhashable regardless. Configs are compared and copied by value via ``model_dump``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    side: Side
    view: str | None = None
    featurizers: tuple[FeaturizerConfig, ...] | None = None
    options: FrozenMapping | None = None
    hyperparameter_space: FrozenMapping | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_recipe_input(cls, data: object) -> object:
        """Rewrite any accepted spelling of a featurizer into this model's own fields.

        :param data: A mapping of fields, ``side`` included.
        :returns: Canonical field mapping, or *data* unchanged if it is not a mapping or list.
        """
        if not isinstance(data, (list, dict)):
            return data
        return normalize_featurizer(data)

    @classmethod
    def parse(cls, raw: object, side: Side | str | None) -> FeaturizerConfig:
        """Validate a featurizer config in any accepted spelling for *side*.

        :param raw: A recipe string (``name``, ``name[view]``, ``a+b``), a mapping of fields or a
            ``FeaturizerConfig``.
        :param side: The side the featurizer is used on; ``None`` takes the side *raw* sets.
        :returns: The validated config.
        :raises ValueError: If *raw* is a config of the other side, or does not validate.
        """
        if isinstance(raw, FeaturizerConfig):
            if side is not None and raw.side != side:
                raise ValueError(f"featurizer {raw.name!r} has side {raw.side.value!r} but is used on the {side} side")
            return raw
        return cls.model_validate(normalize_featurizer(raw, None if side is None else str(side)))

    @property
    def featurizer_class(self) -> type[Featurizer]:
        """The registered featurizer class this node names on its side."""
        return get_featurizer(self.name, self.side)

    @model_validator(mode="after")
    def _validate_hyperparameter_space(self) -> FeaturizerConfig:
        """Check this node's tuning search space against the shared space schema.

        Catches a malformed space at config load time rather than deep inside a tuning
        run. The featurizer name is passed as context so the error names the offender.

        :returns: This config, unchanged.
        """
        if self.hyperparameter_space is not None:
            validate_hyperparameter_space(
                self.hyperparameter_space,
                context=f"FeaturizerConfig({self.name!r}).hyperparameter_space",
            )
        return self

    @model_validator(mode="after")
    def _require_non_empty_view(self) -> FeaturizerConfig:
        """Reject a blank ``view``.

        A whitespace-only view would otherwise reach the registry and fail much later
        with a far less obvious error.

        :returns: This config, unchanged.
        :raises ValueError: If ``view`` is set but blank.
        """
        if self.view is not None and not str(self.view).strip():
            msg = "view must be a non-empty string when set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_explicit_view_for_parametric_featurizers(self) -> FeaturizerConfig:
        """Require a view for featurizers that are only meaningful per view.

        Such featurizers are addressed with a parametric label like ``pca[expression]``,
        so a bare name is ambiguous rather than defaultable.

        :returns: This config, unchanged.
        :raises ValueError: If the featurizer needs an explicit view and none is set.
        """
        if requires_explicit_view(self.name, self.side) and not self.view:
            msg = f"Featurizer {self.name!r} requires an explicit view, e.g. {self.name}[gene_expression]"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_concat_children(self) -> FeaturizerConfig:
        """Confine nested ``featurizers`` to ``concatFeaturizers``, which must have some.

        Keeps the tree shape honest: only the concat node combines children, and an empty
        concat node would build no features at all.

        :returns: This config, unchanged.
        :raises ValueError: If concat has no children, or a non-concat featurizer has any.
        """
        if self.name == CONCAT:
            if not self.featurizers:
                msg = "concatFeaturizers requires a non-empty featurizers list"
                raise ValueError(msg)
            other = [child.name for child in self.featurizers if child.side != self.side]
            if other:
                msg = f"concatFeaturizers on the {self.side} side cannot hold featurizers of the other side: {other}"
                raise ValueError(msg)
        elif self.featurizers is not None:
            msg = f"Featurizer {self.name!r} does not accept nested featurizers"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_unique_selectors(self) -> FeaturizerConfig:
        """Reject a concat tree that repeats the same leaf selector, e.g. ``view[gene_expression]``.

        Duplicates would silently emit the same feature block twice. The same base name on
        different views is fine.

        :returns: This config, unchanged.
        :raises ValueError: If the same selector appears twice.
        """
        seen: set[str] = set()
        for leaf in self.leaves():
            if leaf.selector in seen:
                msg = (
                    f"Duplicate featurizer selector {leaf.selector!r} in one featurizer slot. "
                    "Each qualified featurizer may appear at most once per slot."
                )
                raise ValueError(msg)
            seen.add(leaf.selector)
        return self

    @property
    def selector(self) -> str:
        """This node in hyperparameter keys and concat blocks: ``name[view]``, or ``name`` without a view."""
        return self.name if self.view is None else f"{self.name}[{self.view}]"

    def leaves(self) -> Iterator[FeaturizerConfig]:
        """Yield the leaf featurizers of this tree, expanding concat nodes in order.

        :yields: This node when it is a leaf, else the leaves of its children.
        """
        if self.name != CONCAT:
            yield self
            return
        for child in self.featurizers or ():
            yield from child.leaves()

    def leaf_kwargs(self) -> dict[str, Any]:
        """Return this leaf's construction kwargs at default hyperparameters.

        The fixed ``options``, then the ``default`` of every space entry (the config's own space,
        else the featurizer class's), then the ``view``. These are the kwargs that
        ``Featurizer.output_blocks`` and ``Featurizer.views`` read.

        :returns: Keyword arguments for featurizer construction.
        """
        kwargs: dict[str, Any] = dict(self.options or {})
        space = self.hyperparameter_space or self.featurizer_class.get_hyperparameter_space()
        for key, spec in space.items():
            if isinstance(spec, Mapping) and "default" in spec:
                kwargs.setdefault(key, spec["default"])
        if self.view is not None:
            kwargs.setdefault("view", self.view)
        return kwargs

    def views(self) -> list[str]:
        """Return the raw views the leaves of this tree read, at default hyperparameters.

        :returns: View names, leaf by leaf.
        """
        return [view for leaf in self.leaves() for view in leaf.featurizer_class.views(leaf.leaf_kwargs())]

    def output_blocks(self) -> tuple[BlockSpec, ...]:
        """Return the blocks this tree emits, at default hyperparameters: the leaves' blocks in order.

        :returns: Block specs emitted by the featurizer tree.
        """
        return tuple(spec for leaf in self.leaves() for spec in leaf.featurizer_class.output_blocks(leaf.leaf_kwargs()))

    def create_instance(self, hyperparameters: Mapping[str, Any] | None = None) -> Featurizer:
        """Instantiate the configured featurizer from the registry.

        Turns this declarative template into a live object: resolves ``name`` in the
        featurizer registry, then merges constructor arguments so that
        *hyperparameters* (typically a tuning trial's picks) override the config's own
        ``options``. ``view`` and ``featurizers`` are filled in only when the
        caller has not already supplied them.

        :param hyperparameters: Concrete constructor values for this node. Nested
            concat children should already be resolved by the caller into instances
            or config payloads under the ``featurizers`` key.
        :returns: Featurizer instance for this config.
        """
        cls = self.featurizer_class
        hp = thaw_value(dict(self.options or {}))
        hp.update(thaw_value(dict(hyperparameters or {})))
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.featurizers is not None and "featurizers" not in hp:
            hp["featurizers"] = [child.model_dump(mode="python") for child in self.featurizers]
        return cls(**hp)


FeaturizerConfig.model_rebuild()
