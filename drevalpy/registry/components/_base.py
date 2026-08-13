"""ComponentRegistry -- shared base for predictor and featurizer registries.

Adds contract assignment, metadata/tags/reference handling, and
register_existing() on top of the abstract Registry base.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureContract, normalize_feature_contract
from drevalpy.log import get_logger
from drevalpy.registry._base import Registry
from drevalpy.registry.components._abstract import validate_no_abstract_methods
from drevalpy.registry.components._metadata_validate import validate_registered_class
from drevalpy.registry.components._registration_metadata import (
    RegistrationMetadata,
    normalize_registration_metadata,
)
from drevalpy.types.enums.literature_reference import LiteratureReference

logger = get_logger(__name__)


class ComponentRegistry(Registry):
    """Base for registries that attach contracts, tags, and literature references."""

    _required_fields: ClassVar[tuple[str, ...]] = ("description",)

    def _normalize_metadata(
        self,
        description: str,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
    ) -> RegistrationMetadata:
        """Validate and normalize shared registration kwargs.

        :param description: Short human-readable summary.
        :param tags: Optional discovery tags.
        :param reference: Optional literature citation metadata.
        :returns: Frozen metadata object.
        """
        return normalize_registration_metadata(description, tags, reference)

    def _apply_metadata(self, cls: type[Any], metadata: RegistrationMetadata) -> None:
        """Attach normalized registration metadata to *cls*.

        :param cls: Class receiving registration metadata.
        :param metadata: Normalized description, tags, and optional reference.
        """
        cls.description = metadata.description
        cls.tags = metadata.tags
        cls.reference = metadata.reference

    def _apply_contract(
        self,
        cls: type[Any],
        attr_name: str,
        contract: FeatureContract | None,
    ) -> None:
        """Assign a normalized contract attribute, preferring the decorator argument.

        A class-body declaration is a valid way to state a contract: it is used
        whenever the registration decorator was given none. When both are present
        the decorator wins, since it is the more specific of the two.

        :param cls: Class being registered.
        :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
        :param contract: Already-normalized feature contract from the decorator, or
            ``None`` to fall back to the class declaration.
        :raises ValueError: If neither the decorator nor the class declares *attr_name*.
        """
        if contract is None:
            contract = self._declared_contract(cls, attr_name)
        elif attr_name in cls.__dict__:
            logger.debug(
                "%s: decorator %s overrides the class-body declaration",
                cls.__name__,
                attr_name,
            )
        setattr(cls, attr_name, contract)

    def _declared_contract(self, cls: type[Any], attr_name: str) -> FeatureContract:
        """Return the contract *cls* declares under *attr_name*.

        :param cls: Class being registered.
        :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
        :returns: Normalized contract taken from the class declaration.
        :raises ValueError: If the class declares no usable contract.
        """
        declared = getattr(cls, attr_name, None)
        if declared is None:
            msg = (
                f"{cls.__name__}: no {attr_name} declared; pass {attr_name}= to the "
                f"registration decorator or set it on the class body"
            )
            raise ValueError(msg)
        try:
            return normalize_feature_contract(declared)
        except TypeError as exc:
            msg = f"{cls.__name__}: class-body {attr_name} is invalid: {exc}"
            raise ValueError(msg) from exc

    def register_existing(self, name: str, cls: type[Any]) -> None:
        """Register a class that was previously decorated but removed via ``clear``.

        :param name: Registry name under which *cls* should be restored.
        :param cls: Component class with registration metadata attributes.
        """
        with self._lock:
            if name in self._store:
                return
            validate_registered_class(
                self._registry_id,
                name,
                cls,
                required_fields=self._required_fields,
            )
            self._validate_registration(name, cls)
            self._store[name] = cls
            cls.registry_name = name
            logger.debug("Registered %s: %s", self._label, name)

    def _validate_registration(self, name: str, cls: type[Any]) -> None:
        """Run registry-specific class invariants after metadata validation.

        Subclasses that override this must call ``super()`` so the shared
        abstract-member check keeps running.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Component class being registered or restored.
        """
        validate_no_abstract_methods(self._registry_id, name, cls)
