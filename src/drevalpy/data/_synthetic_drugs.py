"""Drug-side views and ``uns`` payloads of :func:`drevalpy.data.synthetic`.

Only *raw* data is authored: the SMILES. Everything a featurizer can derive from
them is derived by the real featurizer, so the dataset cannot drift away from what
the library computes: Morgan fingerprints and molecular graphs, and the BPE merges
learned by ``subword-nmt``. The views that come from pretrained weights in
production (``chemberta``, ``smilesvec``, ``bpe_smiles``) are seeded noise of the
correct width. ``rdkit``, ``torch_geometric`` and ``subword-nmt`` are imported
inside the functions that need them, because :mod:`drevalpy.data` sits on the
critical path of ``import drevalpy``.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from drevalpy.data._synthetic_omics import check_choices

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset

#: Real, rdkit-parseable SMILES, so the ``fingerprints`` featurizer works. The
#: simulated responses do not depend on them.
SMILES: Final = (
    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
    "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "CS(=O)(=O)CCNCC1=CC2=C(C=C1)N=CN=C2NC3=CC(=C(C=C3)OCC4=CC(=CC=C4)F)Cl",
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    "CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C",
    "CC1=NC(=CC(=N1)N2CCN(CC2)CCO)NC3=NC=C(S3)C(=O)NC4=C(C=CC=C4Cl)C",
    "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4",
    "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F",
    "C1=C(C(=O)NC(=O)N1)F",
    "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3",
    "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O",
    "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO",
    "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O",
    "C1=NC2=C(N1)C(=S)N=CN2",
    "NC(=O)NO",
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
    "CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O",
    "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
    "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(C=C1)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN(C)C(=N)N=C(N)N",
    "CN1CCCC1C2=CN=CC=C2",
)

#: Most drugs :func:`synthetic` can build; one SMILES per drug.
MAX_SYNTHETIC_DRUGS: Final = len(SMILES)

#: Most drugs :func:`synthetic` can build; one SMILES per drug.
MAX_SYNTHETIC_DRUGS: Final = len(SMILES)

FINGERPRINT_BITS: Final = 128
CHEMBERTA_DIM: Final = 768
BPE_LENGTH: Final = 128
SMILESVEC_DIM: Final = 100

#: Names accepted by ``drug_views``, stored in ``response.varm``. The fingerprint is
#: computed; the rest are seeded noise of the width the production featurizer emits.
DRUG_VIEWS: Final = ("morgan_fingerprint", "chemberta", "bpe_smiles", "smilesvec")
_NOISE_WIDTH: Final = {"chemberta": CHEMBERTA_DIM, "bpe_smiles": BPE_LENGTH, "smilesvec": SMILESVEC_DIM}

#: Names accepted by ``uns_payloads``, stored in ``mdata.uns`` as the published files do.
UNS_PAYLOADS: Final = ("drug_graphs", "pathways_gmt", "sparsego", "bpe_codes")


def _pathways_gmt(genes: Sequence[str]) -> str:
    """Build a two-set GMT string; GSVA hard-codes ``min_size=5``."""
    half = max(5, len(genes) // 2)
    sets = {"SYNTH_PATHWAY_A": genes[:half], "SYNTH_PATHWAY_B": genes[-half:]}
    return "".join("\t".join([name, "synthetic", *members]) + "\n" for name, members in sets.items())


def _sparsego_uns(genes: Sequence[str]) -> dict[str, str]:
    """Build ``uns['sparsego']`` in the two-file text form the SparseGO reader takes."""
    terms = ("GO:0006259", "GO:0008283")
    ontology_rows = [f"{terms[0]}\t{terms[1]}\tdefault"]
    for index, gene in enumerate(genes):
        ontology_rows.append(f"{terms[index % len(terms)]}\t{gene}\tgene")
    return {
        "gene2ind": "".join(f"{index}\t{gene}\n" for index, gene in enumerate(genes)),
        "ontology": "".join(row + "\n" for row in ontology_rows),
    }


def _bpe_codes(smiles: Sequence[str], *, num_symbols: int = 50) -> str:
    """Learn real BPE merges from the dataset SMILES.

    ``uns['bpe_codes']`` is not read by any library code -- the PharmaFormer
    featurizer relearns merges at fit time -- but the published datasets carry
    it. Learning them for real also proves the SMILES are dense enough for
    ``subword-nmt``, which refuses to merge when no character pair occurs twice.

    :param smiles: SMILES strings to learn merges from.
    :param num_symbols: Number of merge operations to learn.
    :returns: BPE codes file contents.
    """
    from subword_nmt.learn_bpe import learn_bpe

    codes = io.StringIO()
    learn_bpe(io.StringIO("\n".join(smiles) + "\n"), codes, num_symbols=num_symbols, verbose=False)
    return codes.getvalue()


def _graph_to_dict(graph: Any) -> dict[str, np.ndarray]:
    """Store a graph as plain arrays, which is what ``h5py`` round-trips."""
    return {
        "x": np.asarray(graph.x, dtype=np.float32),
        "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
        "edge_attr": np.asarray(graph.edge_attr, dtype=np.float32),
    }


def _drug_graphs(dataset: Dataset) -> dict[str, dict[str, np.ndarray]]:
    """Featurize every drug's SMILES into a molecular graph with the real featurizer."""
    from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
    from drevalpy.types.data.feature_source import DrugFeatureSource

    drug_ids = dataset.drug_ids
    graphs = DrugGraphFeaturizer()._compute_from_source(DrugFeatureSource(dataset, drug_ids), drug_ids)
    return {
        str(drug_id): _graph_to_dict(graph)
        for drug_id, graph in zip(drug_ids, graphs, strict=True)
        if graph is not None
    }


def _fingerprints(dataset: Dataset) -> np.ndarray:
    """Featurize every drug's SMILES into Morgan fingerprints with the real featurizer."""
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
    from drevalpy.types.data.feature_source import DrugFeatureSource

    drug_ids = dataset.drug_ids
    featurizer = FingerprintsFeaturizer(n_bits=FINGERPRINT_BITS)
    return featurizer._compute_from_source(DrugFeatureSource(dataset, drug_ids), drug_ids)


def add_drug_views(dataset: Dataset, views: Sequence[str], rng: np.random.Generator) -> None:
    """Fill ``response.varm`` with the requested drug views.

    :param dataset: Dataset modified in place; its drugs need ``canonical_smiles``.
    :param views: Names from :data:`DRUG_VIEWS`.
    :param rng: Source of the noise views.
    :raises ValueError: If a name is not in :data:`DRUG_VIEWS`.
    """
    check_choices("drug_views", views, DRUG_VIEWS)
    response = dataset.mdata.mod["response"]
    for view in views:
        if view == "morgan_fingerprint":
            response.varm[view] = _fingerprints(dataset)
        else:
            response.varm[view] = rng.normal(size=(response.n_vars, _NOISE_WIDTH[view])).astype(np.float32)


def add_uns_payloads(dataset: Dataset, payloads: Sequence[str], genes: Sequence[str]) -> None:
    """Fill ``mdata.uns`` with the auxiliary payloads published datasets carry.

    :param dataset: Dataset modified in place; its drugs need ``canonical_smiles``.
    :param payloads: Names from :data:`UNS_PAYLOADS`.
    :param genes: Gene symbols the pathway and ontology payloads are written over.
    :raises ValueError: If a name is not in :data:`UNS_PAYLOADS`.
    """
    check_choices("uns_payloads", payloads, UNS_PAYLOADS)
    uns = dataset.mdata.uns
    for payload in payloads:
        if payload == "drug_graphs":
            uns[payload] = _drug_graphs(dataset)
        elif payload == "pathways_gmt":
            uns[payload] = _pathways_gmt(genes)
        elif payload == "sparsego":
            uns[payload] = _sparsego_uns(genes)
        else:
            uns[payload] = _bpe_codes(list(dataset.mdata.mod["response"].var["canonical_smiles"]))
