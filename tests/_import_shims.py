"""Make an optional third-party import fail, to exercise the guidance it raises.

Every featurizer that reaches for an optional dependency inside a method turns an
``ImportError`` into an actionable message naming the extra to install. Pinning
that message means faking the missing package, and the only lever that reaches an
import statement inside an already-imported module is ``builtins.__import__``.
Nine tests had each written out the same shim.

Sibling of :mod:`tests._barrel_surface` and :mod:`tests._trusted_subprocess`: an
``_``-prefixed module at the ``tests/`` root, imported as
``from tests._import_shims import block_imports``. The underscore keeps it out of
collection, and the mirror policy walks ``drevalpy/`` only, so no mirrored test is
demanded for it.
"""

from __future__ import annotations

import builtins

import pytest


def block_imports(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    """Raise ``ImportError`` for any module whose name starts with a given prefix.

    Patches ``builtins.__import__`` rather than deleting from ``sys.modules``,
    because the module under test imports the optional dependency inside the
    function being called - by then a ``sys.modules`` edit is too late, and an
    already-imported package would be found again.

    :param monkeypatch: Pytest fixture; the patch is undone at teardown.
    :param prefixes: Module-name prefixes to reject. ``"torch"`` rejects
        ``torch_geometric`` too, which is what every caller wants.
    """
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name.startswith(prefixes):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
