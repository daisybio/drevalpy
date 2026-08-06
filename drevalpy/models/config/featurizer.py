"""Declarative featurizer configuration schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.components.featurizer_label import requires_explicit_view
from drevalpy.components.featurizer_tree import ensure_unique_qualified_featurizers
from drevalpy.components.hyperparameter_space import validate_hyperparameter_space
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer
from drevalpy.models.config._featurizer_parse import normalize_featurizer_config
from drevalpy.models.config._recipe import expand_featurizer_recipe
from drevalpy.models.config.immutable import FrozenMapping, thaw_value


class FeaturizerConfig(BaseModel):
    """Immutable template for a featurizer node in a model stack.

    Describes *which* featurizer to build and how to address it, not the built object:
    a ``name`` looked up in one of the two registries (``cell_line`` or ``drug``), an
    optional ``view`` selecting the input matrix, ``options`` fixing concrete
    constructor values, and ``hyperparameter_space`` declaring what tuning may vary.
    ``featurizers`` makes the node a tree, holding children for ``concatFeaturizers``.
    Combine several views by nesting one single-``view`` child per view under a concat
    node, which is what the ``raw[expression]+raw[mutations]`` shorthand expands to.

    Accepts the same notations users write in the docs' recipe strings and YAML (a bare
    name, a ``name[view]`` label, a list, or a one-key mapping) and normalizes them into
    these fields. Validation is front-loaded here so a bad recipe fails at load time
    instead of mid-run.

    ``tuple`` fields plus ``frozen=True`` make an accidental in-place edit of a shared or
    cached config fail loudly. Note this buys *safety*, not hashability: ``options`` and
    ``hyperparameter_space`` hold arbitrary nested data, so a config carrying either is
    unhashable regardless. Configs are compared and copied by value via ``model_dump``.

    Subclasses pin ``registry`` to a single value; see ``CellLineFeaturizerConfig``
    and ``DrugFeaturizerConfig``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    registry: Literal["cell_line", "drug"] = "cell_line"
    view: str | None = None
    featurizers: tuple[FeaturizerConfig, ...] | None = None
    options: FrozenMapping | None = None
    hyperparameter_space: FrozenMapping | None = None

    @classmethod
    def _pinned_registry(cls) -> str | None:
        """Return the single registry this class is locked to, if any.

        Subclasses narrow ``registry`` to a one-value ``Literal``, so that annotation is
        the only place each registry name is declared.

        :returns: The sole allowed ``registry`` value, or ``None`` when several are allowed.
        """
        allowed = get_args(cls.model_fields["registry"].annotation)
        return str(allowed[0]) if len(allowed) == 1 else None

    @model_validator(mode="before")
    @classmethod
    def _normalize_recipe_input(cls, data: object) -> object:
        """Rewrite the various ways of writing one featurizer into this model's own fields.

        A featurizer can be given as a compact recipe string or spelled out as a mapping.
        This runs before field validation and reduces every accepted form to the canonical
        ``name`` / ``view`` / ``featurizers`` fields::

            "scaledGeneExpression"                     name=scaledGeneExpression
            "raw[gene_expression]"                     name=raw, view=gene_expression
            "raw[gene_expression]+raw[mutations]"      name=concatFeaturizers, two children
            ["raw[gene_expression]", "raw[mutations]"] name=concatFeaturizers, two children
            {"pca[methylation]": {"n_components": 8}}  name=pca, view=methylation, + space
            {"name": "raw", "view": "gene_expression"} already canonical, passed through

        A recipe string is expanded into the mapping it stands for first, so the mapping
        normalizer sees the same input whichever notation was used. Turning a bare name into a
        component needs a registry to look it up in. A pinned subclass uses its own and
        overwrites a conflicting ``registry`` key; the base class reads it from the input,
        falling back to the field default.

        :param data: A recipe string, a list of them, or a mapping of fields.
        :returns: Canonical field mapping, or *data* unchanged if it is none of these forms.
        """
        if not isinstance(data, (str, list, dict)):
            return data
        if isinstance(data, str):
            data = expand_featurizer_recipe(data)
        pinned = cls._pinned_registry()
        if pinned is not None:
            if isinstance(data, dict) and "registry" in data:
                data = {**data, "registry": pinned}
            return normalize_featurizer_config(data, default_registry=pinned)
        requested = data.get("registry") if isinstance(data, dict) else None
        fallback = cls.model_fields["registry"].default
        return normalize_featurizer_config(data, default_registry=str(requested or fallback))

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
        if requires_explicit_view(self.name) and not self.view:
            msg = f"Featurizer {self.name!r} requires an explicit view, e.g. {self.name}[expression]"
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
        if self.name == "concatFeaturizers":
            if not self.featurizers:
                msg = "concatFeaturizers requires a non-empty featurizers list"
                raise ValueError(msg)
        elif self.featurizers is not None:
            msg = f"Featurizer {self.name!r} does not accept nested featurizers"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_unique_qualified_children(self) -> FeaturizerConfig:
        """Reject a concat tree that repeats the same qualified leaf, e.g. ``raw[expression]``.

        Duplicates would silently emit the same feature block twice. The same base name on
        different views is fine.

        :returns: This config, unchanged.
        """
        if self.name != "concatFeaturizers":
            return self
        ensure_unique_qualified_featurizers(self, str(self.registry))
        return self

    def create_instance(self, hyperparameters: Mapping[str, Any] | None = None):
        """Instantiate the configured featurizer from the registry.

        Turns this declarative template into a live object: resolves ``name`` in the
        registry named by ``registry``, then merges constructor arguments so that
        *hyperparameters* (typically a tuning trial's picks) override the config's own
        ``options``. ``view`` and ``featurizers`` are filled in only when the
        caller has not already supplied them.

        :param hyperparameters: Concrete constructor values for this node. Nested
            concat children should already be resolved by the caller into instances
            or config payloads under the ``featurizers`` key.
        :returns: Featurizer instance for this config.
        """
        if self.registry == "cell_line":
            cls = get_cell_line_featurizer(self.name)
        else:
            cls = get_drug_featurizer(self.name)
        hp = thaw_value(dict(self.options or {}))
        hp.update(thaw_value(dict(hyperparameters or {})))
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.featurizers is not None and "featurizers" not in hp:
            hp["featurizers"] = [child.model_dump(mode="python") for child in self.featurizers]
        return cls(**hp)


class CellLineFeaturizerConfig(FeaturizerConfig):
    """Featurizer config fixed to the cell-line registry.

    Use in a slot that must hold cell-line features: a mismatched ``registry`` in the
    payload is corrected to ``cell_line`` rather than accepted.
    """

    registry: Literal["cell_line"] = "cell_line"


class DrugFeaturizerConfig(FeaturizerConfig):
    """Featurizer config fixed to the drug registry.

    Drug-side counterpart of ``CellLineFeaturizerConfig``.
    """

    registry: Literal["drug"] = "drug"


FeaturizerConfig.model_rebuild()
CellLineFeaturizerConfig.model_rebuild()
DrugFeaturizerConfig.model_rebuild()
