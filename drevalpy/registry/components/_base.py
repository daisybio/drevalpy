"""ComponentRegistry -- shared base for predictor and featurizer registries.

Adds contract assignment, metadata/tags/reference handling, and
register_existing() on top of the abstract Registry base.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.log import get_logger
from drevalpy.registry._base import Registry
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

    def _apply_contract(self, cls: type[Any], attr_name: str, contract: FeatureContract) -> None:
        """Assign a normalized contract attribute, rejecting class-body definitions.

        :param cls: Class being registered.
        :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
        :param contract: Already-normalized feature contract to attach.
        :raises ValueError: If *attr_name* is already defined on the class body.
        """
        if attr_name in cls.__dict__:
            msg = (
                f"{cls.__name__}: do not set {attr_name} on the class body; "
                "pass it to the registration decorator instead"
            )
            raise ValueError(msg)
        setattr(cls, attr_name, contract)

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

        Override in subclasses to add custom checks.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Component class being registered or restored.
        """
        return
