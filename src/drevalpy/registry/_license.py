"""SPDX license expressions of components, and the license a component inherits from its package.

``license_expression`` is imported inside the functions and its SPDX index is built once.
"""

from __future__ import annotations

import importlib.metadata
from functools import cache
from typing import Any

#: The SPDX value for "no license information".
NOASSERTION = "NOASSERTION"


@cache
def _licensing() -> Any:
    from license_expression import get_spdx_licensing

    return get_spdx_licensing()


def _is_custom_key(key: str) -> bool:
    return key == NOASSERTION or key.startswith("LicenseRef-")


def spdx(value: str, *, field: str) -> str:
    """Return *value* as a normalized SPDX license expression.

    Known license ids take their canonical case (``"mit"`` becomes ``"MIT"``). ``NOASSERTION``
    and ``LicenseRef-*`` ids are accepted as they are.

    :param value: An SPDX expression such as ``"MIT"`` or ``"MIT OR Apache-2.0"``.
    :param field: Name of the argument, for the message.
    :returns: The normalized expression.
    :raises TypeError: If *value* is not a string.
    :raises ValueError: If *value* is blank, does not parse, or names an unknown license id.
    """
    from license_expression import ExpressionError, ExpressionParseError

    if not isinstance(value, str):
        msg = f"{field} must be an SPDX license expression string, got {type(value).__name__}"
        raise TypeError(msg)
    if not value.strip():
        msg = f"{field} must be a non-empty SPDX license expression, such as 'MIT'"
        raise ValueError(msg)
    licensing = _licensing()
    try:
        expression = licensing.parse(value.strip())
    except (ExpressionError, ExpressionParseError) as exc:
        msg = f"{field} {value!r} is not a valid SPDX license expression: {exc}"
        raise ValueError(msg) from None
    unknown = [key for key in licensing.unknown_license_keys(expression) if not _is_custom_key(key)]
    if unknown:
        msg = (
            f"{field} {value!r} names unknown SPDX license id(s): {', '.join(unknown)}. "
            "Use an id from https://spdx.org/licenses/, a LicenseRef-..., or NOASSERTION."
        )
        raise ValueError(msg)
    return str(expression.render())


def _distribution(top_level: str) -> importlib.metadata.Distribution | None:
    names = importlib.metadata.packages_distributions().get(top_level, [])
    for name in [*names, top_level]:
        try:
            return importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@cache
def _package_license(top_level: str) -> str:
    dist = _distribution(top_level)
    if dist is None:
        return NOASSERTION
    for key in ("License-Expression", "License"):
        value = dist.metadata.get(key)
        if value:
            try:
                return spdx(value, field=key)
            except ValueError:
                continue
    return NOASSERTION


def distribution_license(cls: type[Any]) -> str:
    """Return the license of the installed package that defines *cls*.

    The package is the distribution of the top-level module of ``cls.__module__``. Its
    ``License-Expression`` is used, else its legacy ``License`` field when that is a valid SPDX
    expression, else ``NOASSERTION``. A class from a file that is not part of an installed
    package (an extension file, a test module) gets ``NOASSERTION``.

    :param cls: A component class.
    :returns: An SPDX license expression.
    """
    return _package_license(cls.__module__.partition(".")[0])
