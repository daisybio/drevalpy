"""Drug featurizers."""

from __future__ import annotations

import importlib

_SUBMODULES = (
    "bpe_pharmaformer",
    "chemberta",
    "concat",
    "drug_graph",
    "fingerprints",
    "identity",
    "molgnet",
    "smilesvec",
    "view",
)

for _submodule in _SUBMODULES:
    importlib.import_module(f"{__name__}.{_submodule}")
