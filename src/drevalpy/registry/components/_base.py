"""ComponentRegistry -- shared base for predictor and featurizer registries.

Adds ``register_existing()`` and the registry-wide class invariants on top of the
abstract ``Registry`` base. The two concerns that used to sit here but read no
registry state - resolving a contract and stamping registration metadata onto a
class - live in ``_contract_assignment.py`` and ``_registration_metadata.py``, and
the concrete registries call those directly.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.log import get_logger
from drevalpy.registry._base import Registry
from drevalpy.registry.components._abstract import validate_no_abstract_methods
from drevalpy.registry.components._metadata_validate import validate_registered_class

logger = get_logger(__name__)


class ComponentRegistry(Registry):
    """Base for registries whose entries carry validated registration metadata."""

    _required_fields: ClassVar[tuple[str, ...]] = ("description",)

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
