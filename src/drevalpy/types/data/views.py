"""Stored MuData modality names behind the public view names.

Configs, recipes and predictor code address a sample view by its public name. The
modality actually stored in the published ``.h5mu`` files can differ:
``copy_number_variation_gistic`` is the public name, but the datasets store the
modality as ``copy_number_variation``. Every read of ``MuData.mod`` by a
caller-supplied view goes through :func:`find_view`, which prefers the name as
written, so files that store the public name (as users building their own dataset
are told to) are read too. :func:`stored_view_name` is the name written into new
files and :func:`public_view_name` its inverse, for messages and listings. All three
pass a name that is not in the map through unchanged, so custom matrices and
non-omics views keep working.

The constants below are the one spelling of every other storage name a reader of the
MuData needs: the ``obs`` and ``var`` columns and the ``uns`` keys.
"""

from __future__ import annotations

from collections.abc import Container
from types import MappingProxyType
from typing import Final

#: Column of the sample metadata (``mdata.obs``) holding the tissue label; also its view name.
TISSUE_IDENTIFIER: Final = "tissue"
#: Column of ``response.var`` holding each drug's canonical SMILES; also its view name.
SMILES_COLUMN: Final = "canonical_smiles"
#: View name of the drug graphs, which are stored under :data:`DRUG_GRAPHS_KEY`.
DRUG_GRAPH_VIEW: Final = "drug_graph"
#: ``mdata.uns`` key of the drug graphs, ``{drug_id: {"x", "edge_index", "edge_attr"}}``.
DRUG_GRAPHS_KEY: Final = "drug_graphs"
#: ``mdata.uns`` key of the pathway gene sets, as GMT text.
PATHWAYS_GMT_KEY: Final = "pathways_gmt"
#: ``mdata.uns`` key of the SparseGO ontology mapping.
SPARSEGO_KEY: Final = "sparsego"
#: Key in the ``uns`` of an expression modality naming its unit, such as ``log2_tpm_plus_1``.
EXPRESSION_UNIT_KEY: Final = "expression_unit"

_STORED_NAMES: Final = MappingProxyType({"copy_number_variation_gistic": "copy_number_variation"})
_PUBLIC_NAMES: Final = MappingProxyType({stored: public for public, stored in _STORED_NAMES.items()})


def stored_view_name(view: str) -> str:
    """Translate a public view name into the modality name stored in the MuData.

    :param view: Public view name, e.g. ``"copy_number_variation_gistic"``.
    :returns: The modality name to look up in ``MuData.mod``.
    """
    return _STORED_NAMES.get(view, view)


def public_view_name(stored: str) -> str:
    """Translate a stored modality name back into the view name users write.

    :param stored: Modality name as stored in the MuData.
    :returns: The public view name, or *stored* unchanged if it has none.
    """
    return _PUBLIC_NAMES.get(stored, stored)


def find_view(view: str, available: Container[str]) -> str | None:
    """Pick the stored modality that holds *view*.

    :param view: Public view name, e.g. ``"copy_number_variation_gistic"``.
    :param available: Modality names present in the MuData.
    :returns: *view* itself if stored under that name, else its :func:`stored_view_name` if
        stored, else ``None``.
    """
    for name in (view, stored_view_name(view)):
        if name in available:
            return name
    return None
