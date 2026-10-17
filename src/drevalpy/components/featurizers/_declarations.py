"""What a featurizer declares: its contract, the views it reads, the blocks it emits.

These are the class-body declarations the registry and the config layer read
*without instantiating anything* - the contract normalization that runs at class
creation, the view resolution the model config calls to know what to load from
disk, and the output block specs ``models/config/_block_specs.py`` reads to
predict a featurizer's output shape. None of it touches fitted state, which is
what separates it from the fit/transform contract in ``base.py``.

The module is ``_``-prefixed so ``_discover_modules`` in
``drevalpy/registry/_builtins.py`` does not scan it as a component.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import (
    FeatureContract,
    featurizer_contract,
    normalize_feature_contract,
)
from drevalpy.types.data.batch.feature_block import BlockSpec


class FeaturizerDeclarationsMixin:
    """Hold and resolve a featurizer's class-body declarations."""

    contract: ClassVar[FeatureContract]
    precompute: ClassVar[bool] = False
    requires_view: ClassVar[bool] = False
    entity_id_only: ClassVar[bool] = False
    input_views: ClassVar[tuple[str, ...] | None] = None
    source_views: ClassVar[tuple[str, ...] | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Normalize a class-body ``contract`` declaration, if there is one.

        A ``FeatureFormat`` shorthand is widened to a ``FeatureContract`` so the
        class body and the ``@register`` argument accept the same spellings.
        Subclasses that declare nothing are registered with the decorator's
        ``contract=`` instead.

        :param kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If a class-body ``contract`` is neither a
            ``FeatureContract`` nor a ``FeatureFormat``.
        """
        super().__init_subclass__(**kwargs)
        if "contract" not in cls.__dict__:
            return
        try:
            cls.contract = normalize_feature_contract(cls.__dict__["contract"])
        except TypeError as exc:
            msg = f"{cls.__name__}: class-body contract is invalid: {exc}"
            raise TypeError(msg) from exc

    @classmethod
    def output_block_specs_for_config(cls, config: Any) -> tuple[BlockSpec, ...]:
        """Return named output blocks for a featurizer config node.

        Declared ``output_block_specs`` win when present; otherwise a single
        block named after the configured (or single declared input) view is emitted.

        :param config: Featurizer config with optional ``view`` / ``hyperparameters``.
        :returns: Block specs emitted by this featurizer under *config*.
        """
        declared = getattr(cls, "output_block_specs", ())
        if declared:
            return tuple(spec for spec in declared if isinstance(spec, BlockSpec))
        view = getattr(config, "view", None)
        if not isinstance(view, str):
            view = cls.input_views[0] if cls.input_views else None
        if isinstance(view, str):
            return (BlockSpec(view, featurizer_contract(cls).format),)
        return ()

    @classmethod
    def resolve_input_views(cls, **kwargs: Any) -> tuple[str, ...]:
        """Return the raw feature views this featurizer reads under *kwargs*.

        An explicit ``view`` kwarg always wins, which covers view-parameterized
        featurizers such as ``raw`` and ``pca``. Otherwise the declared
        ``input_views`` are used. Featurizers whose input depends on other
        hyperparameters override this hook.

        :param kwargs: Featurizer construction / loader kwargs from the model config.
        :returns: Raw view names required from disk, empty when only entity ids are needed.
        :raises TypeError: If the views cannot be determined from *kwargs* and the class body.
        """
        view = kwargs.get("view")
        if isinstance(view, str) and view.strip():
            return (view,)
        if cls.input_views is not None:
            return cls.input_views
        if cls.entity_id_only:
            return ()
        if cls.requires_view:
            msg = f"{cls.__name__} requires an explicit view; pass view= to resolve_input_views"
            raise TypeError(msg)
        msg = (
            f"{cls.__name__}: declare input_views on the class body, set requires_view/entity_id_only, "
            "or override resolve_input_views"
        )
        raise TypeError(msg)
