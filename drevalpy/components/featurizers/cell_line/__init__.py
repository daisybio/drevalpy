"""Cell-line featurizers."""

from __future__ import annotations

import importlib

_SUBMODULES = (
    "bionic",
    "concat",
    "landmark",
    "normalized_proteomics",
    "pathways",
    "pca",
    "raw",
    "scaled_gene_expression",
)

for _submodule in _SUBMODULES:
    importlib.import_module(f"{__name__}.{_submodule}")
