"""Raw-omics MuData factory backing the whole test suite.

The suite used to depend on a gitignored ``data/`` directory holding downloaded
screens, which meant a clean checkout could not run a single test. This module
replaces that with a deterministic, fully in-memory dataset that carries the
same structural slots a published ``.h5mu`` does.

Only *raw* data is authored here: omics matrices, tissue labels and
``canonical_smiles``. Everything a featurizer can derive from rdkit is derived
by the real featurizer, so the fixture cannot drift away from what the library
computes. The remaining views (``chemberta``, ``smilesvec``, ``bpe_smiles``,
``pathway_features``) come from pretrained weights or annotation downloads in
production, so they are filled with seeded noise of the correct width rather
than fetched over the network.

Modality names are taken from :data:`drevalpy.types.data.modalities.OMICS_ACCESSORS`
rather than written as literals, so the fixture always stores omics under the
key the published datasets actually use.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any, Final

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from drevalpy.components.featurizers.cell_line.gene_lists import (
    gene_names_from_list_csv,
    resolve_gene_list_path,
)
from drevalpy.data.utils import CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import OMICS_ACCESSORS, resolve_omics_accessor

#: Roughly the smallest shape that keeps every downstream library happy. At 2x2
#: sklearn emits low-bootstrap warnings and subword-nmt refuses to learn BPE
#: codes ("no pair has frequency >= 2"), both of which are errors under -W.
N_CELL_LINES: Final = 24
N_DRUGS: Final = 8

#: ``LTO`` runs ``KFold(n_splits=2)`` over the unique tissues and then carves a
#: validation tissue out of the training half, so three is the hard floor. Six
#: leaves four cell lines per tissue, which keeps each fold's training half wide
#: enough for the single-drug models.
N_TISSUES: Final = 6

N_GENES: Final = 16
N_METHYLATION_SITES: Final = 20
N_PATHWAYS: Final = 2

FINGERPRINT_BITS: Final = 128
CHEMBERTA_DIM: Final = 768
BPE_LENGTH: Final = 128
SMILESVEC_DIM: Final = 100

#: A curve-metric layer name curation really emits. ``response.X`` holds pEC50 -
#: curation does not duplicate it as a layer - so this one is derived from ``X``.
BUILTIN_MEASURE: Final = "LN_IC50"
RESPONSE_LAYERS: Final = (BUILTIN_MEASURE, "AUC", "IC50")

DATASET_NAME: Final = "SYNTH"
SEED: Final = 20240612

#: Modality key the fixture stores copy-number data under. Resolved through the
#: accessor map so the fixture follows the datasets, not the public alias.
CNV_MODALITY: Final = resolve_omics_accessor("copy_number_variation_gistic")

#: Every omics modality the fixture carries, keyed by the modality name a
#: ``.h5mu`` would use.
OMICS_MODALITIES: Final = tuple(resolve_omics_accessor(name) for name in OMICS_ACCESSORS)

#: Which axis each omics view is measured over, keyed by public omics name.
_VAR_AXIS: Final[Mapping[str, str]] = {
    "gene_expression": "gene",
    "proteomics": "gene",
    "mutations": "gene",
    "methylation": "cpg",
    "copy_number_variation_gistic": "gene",
}

_TISSUES: Final = ("Lung", "Blood", "Skin", "Colon", "Brain", "Breast")

#: Real, rdkit-parseable drug SMILES. Real molecules rather than toy strings so
#: fingerprints, molecular graphs and learned BPE merges are all non-degenerate.
_DRUGS: Final = (
    ("176870", "Erlotinib", "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"),
    ("123631", "Gefitinib", "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"),
    ("208908", "Lapatinib", "CS(=O)(=O)CCNCC1=CC2=C(C=C1)N=CN=C2NC3=CC(=C(C=C3)OCC4=CC(=CC=C4)F)Cl"),
    ("5291", "Imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
    ("216239", "Sorafenib", "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"),
    ("5329102", "Sunitinib", "CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C"),
    ("3062316", "Dasatinib", "CC1=NC(=CC(=N1)N2CCN(CC2)CCO)NC3=NC=C(S3)C(=O)NC4=C(C=CC=C4Cl)C"),
    ("6450551", "Axitinib", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4"),
)

#: Sparse gaps in the response matrix, as ``(cell_line_index, drug_index)``.
#: Kept deliberately thin: the splitters must still leave every tissue and every
#: drug with observed pairs in each fold.
_UNMEASURED_PAIRS: Final = ((0, 3), (5, 1), (9, 6), (14, 0), (18, 5), (22, 7))


def synthetic_gene_symbols(n: int = N_GENES) -> list[str]:
    """Return the first *n* gene symbols shipped in ``landmark_genes.csv``.

    Drawing from the packaged gene list rather than inventing symbols is what
    makes the ``landmarkGenes`` featurizer select a non-empty subset, and means
    the fixture cannot drift out of sync with the shipped lists.

    :param n: Number of symbols to return.
    :returns: Ordered, de-duplicated gene symbols.
    :raises ValueError: If the packaged list holds fewer than *n* symbols.
    """
    symbols = list(dict.fromkeys(gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))))
    if len(symbols) < n:
        msg = f"landmark_genes.csv only provides {len(symbols)} symbols, need {n}"
        raise ValueError(msg)
    return symbols[:n]


def _cell_line_ids() -> np.ndarray:
    return np.array([f"CVCL_S{index:03d}" for index in range(N_CELL_LINES)], dtype=object)


def _tissue_labels() -> np.ndarray:
    return np.array([_TISSUES[index % N_TISSUES] for index in range(N_CELL_LINES)], dtype=object)


def _var_names(axis: str) -> list[str]:
    if axis == "cpg":
        return [
            f"chr{index % 22 + 1}:{1_000_000 + index * 977}-{1_000_800 + index * 977}"
            for index in range(N_METHYLATION_SITES)
        ]
    return synthetic_gene_symbols()


def _omics_matrix(public_name: str, rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    """Draw an omics matrix whose value range is plausible for *public_name*."""
    if public_name == "mutations":
        return rng.binomial(1, 0.25, size=(n_rows, n_cols)).astype(np.float32)
    if public_name == "methylation":
        return rng.beta(2.0, 2.0, size=(n_rows, n_cols)).astype(np.float32)
    if public_name == "copy_number_variation_gistic":
        return rng.integers(-2, 3, size=(n_rows, n_cols)).astype(np.float32)
    return rng.normal(6.0, 1.5, size=(n_rows, n_cols)).astype(np.float32)


def _response_anndata(rng: np.random.Generator, cell_line_ids: np.ndarray) -> ad.AnnData:
    """Build the ``response`` modality: the pair matrix plus drug metadata."""
    drug_ids = np.array([drug[0] for drug in _DRUGS], dtype=object)
    matrix = rng.normal(2.0, 1.0, size=(N_CELL_LINES, N_DRUGS)).astype(np.float32)
    for row, column in _UNMEASURED_PAIRS:
        matrix[row, column] = np.nan

    response = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                CELL_LINE_IDENTIFIER: [f"SYNTH-{index:03d}" for index in range(N_CELL_LINES)],
                TISSUE_IDENTIFIER: _tissue_labels(),
            },
            index=pd.Index(cell_line_ids, name="cellosaurus_id"),
        ),
        var=pd.DataFrame(
            {
                "drug_name": [drug[1] for drug in _DRUGS],
                "canonical_smiles": [drug[2] for drug in _DRUGS],
            },
            index=pd.Index(drug_ids, name="pubchem_id"),
        ),
    )
    response.layers[BUILTIN_MEASURE] = ((6.0 - matrix) * np.log(10.0)).astype(np.float32)
    response.layers["AUC"] = rng.uniform(0.0, 1.0, size=matrix.shape).astype(np.float32)
    response.layers["IC50"] = np.power(10.0, 6.0 - matrix).astype(np.float32)
    # The built-in splitters filter on the CurveCurator quality layers, so a
    # dataset without them cannot be split. Every curve passes here, keeping the
    # folds determined by _UNMEASURED_PAIRS alone.
    for name, layer in _quality_layers(matrix.shape).items():
        response.layers[name] = layer
    return response


def _quality_layers(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Quality layers on which every synthetic curve passes comfortably.

    Values sit far from the thresholds in
    :func:`drevalpy.data.quality.curve_quality_mask`, so a boundary change there
    cannot silently reclassify a synthetic pair.
    """
    passing = {
        "relevance_score": 9.0,
        "fold_change": -2.0,
        "p_value": 1e-9,
        "log_p_value": 9.0,
        "f_value": 400.0,
        "f_value_sam": 80.0,
        "R2": 0.99,
        "RMSE": 0.02,
        "signal_quality": 1.0,
        "slope": 3.0,
        "front": 1.0,
        "back": 0.05,
        "regulation": -1.0,
        # Not filter options, but layers every curated dataset carries: the
        # per-parameter standard errors CurveCurator derives from the fit's
        # Jacobian.
        "pec50_error": 0.05,
        "slope_error": 0.1,
        "front_error": 0.01,
        "back_error": 0.01,
    }
    return {name: np.full(shape, value, dtype=np.float32) for name, value in passing.items()}


def _omics_anndata(
    public_name: str,
    rng: np.random.Generator,
    cell_line_ids: np.ndarray,
    n_covered: int,
) -> ad.AnnData:
    var_names = _var_names(_VAR_AXIS[public_name])
    covered = cell_line_ids[:n_covered]
    return ad.AnnData(
        X=_omics_matrix(public_name, rng, len(covered), len(var_names)),
        obs=pd.DataFrame(index=pd.Index(covered, name="cellosaurus_id")),
        var=pd.DataFrame(index=pd.Index(var_names, name="feature")),
    )


def _pathways_gmt(genes: list[str]) -> str:
    """Build a two-set GMT string; GSVA hard-codes ``min_size=5``."""
    half = max(5, len(genes) // 2)
    sets = {"SYNTH_PATHWAY_A": genes[:half], "SYNTH_PATHWAY_B": genes[-half:]}
    return "".join("\t".join([name, "synthetic", *members]) + "\n" for name, members in sets.items())


def _sparsego_uns(genes: list[str]) -> dict[str, str]:
    """Build ``uns['sparsego']`` in the two-file text form the fixtures use."""
    terms = ("GO:0006259", "GO:0008283")
    ontology_rows = [f"{terms[0]}\t{terms[1]}\tdefault"]
    for index, gene in enumerate(genes):
        ontology_rows.append(f"{terms[index % len(terms)]}\t{gene}\tgene")
    return {
        "gene2ind": "".join(f"{index}\t{gene}\n" for index, gene in enumerate(genes)),
        "ontology": "".join(row + "\n" for row in ontology_rows),
    }


def _bpe_codes(smiles: list[str], *, num_symbols: int = 50) -> str:
    """Learn real BPE merges from the fixture SMILES.

    ``uns['bpe_codes']`` is not read by any library code -- the PharmaFormer
    featurizer relearns merges at fit time -- but the published datasets carry
    it, so the fixture does too. Learning them for real also proves the fixture
    SMILES are dense enough for ``subword-nmt``, which refuses to merge when no
    character pair occurs twice.

    :param smiles: SMILES strings to learn merges from.
    :param num_symbols: Number of merge operations to learn.
    :returns: BPE codes file contents.
    """
    from subword_nmt.learn_bpe import learn_bpe

    codes = io.StringIO()
    learn_bpe(io.StringIO("\n".join(smiles) + "\n"), codes, num_symbols=num_symbols, verbose=False)
    return codes.getvalue()


def _derived_drug_views(dataset: Dataset) -> None:
    """Fill ``response.varm`` and ``uns['drug_graphs']`` from the SMILES column.

    Fingerprints and molecular graphs are produced by the library's own
    featurizers, so the fixture can never disagree with what production
    computes. Graphs are stored as plain dicts of arrays because that is what
    ``h5py`` can round-trip, and what real ``.h5mu`` files contain.
    """
    from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
    from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
    from drevalpy.types.data.feature_source import DrugFeatureSource

    drug_ids = dataset.drug_ids
    source = DrugFeatureSource(dataset, drug_ids)
    response = dataset.mdata.mod["response"]

    fingerprints = FingerprintsFeaturizer(n_bits=FINGERPRINT_BITS)
    response.varm["morgan_fingerprint"] = fingerprints._compute_from_source(source, drug_ids)

    graphs = DrugGraphFeaturizer()._compute_from_source(source, drug_ids)
    dataset.mdata.uns["drug_graphs"] = {
        str(drug_id): _graph_to_dict(graph)
        for drug_id, graph in zip(drug_ids, graphs, strict=True)
        if graph is not None
    }


def _graph_to_dict(graph: Any) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray(graph.x, dtype=np.float32),
        "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
        "edge_attr": np.asarray(graph.edge_attr, dtype=np.float32),
    }


def _pretrained_views(dataset: Dataset, rng: np.random.Generator) -> None:
    """Fill the views that would otherwise need a weight or annotation download."""
    response = dataset.mdata.mod["response"]
    response.varm["chemberta"] = rng.normal(size=(N_DRUGS, CHEMBERTA_DIM)).astype(np.float32)
    response.varm["bpe_smiles"] = rng.normal(size=(N_DRUGS, BPE_LENGTH)).astype(np.float32)
    response.varm["smilesvec"] = rng.normal(size=(N_DRUGS, SMILESVEC_DIM)).astype(np.float32)
    response.obsm["pathway_features"] = rng.normal(size=(N_CELL_LINES, N_PATHWAYS)).astype(np.float32)


def build_synthetic_dataset(
    *,
    name: str = DATASET_NAME,
    omics_coverage: Mapping[str, int] | None = None,
    seed: int = SEED,
) -> Dataset:
    """Build the synthetic raw-omics :class:`~drevalpy.types.data.dataset.Dataset`.

    :param name: Dataset name recorded on the returned object.
    :param omics_coverage: Optional public-omics-name to cell-line-count map.
        Any omics view left out is given full coverage. Reducing a count drops
        trailing cell lines from that modality, which is what makes the
        predictors' NaN-filtering path fire; see
        :mod:`tests.synthetic.variants`.
    :param seed: Seed for every drawn matrix, so the dataset is reproducible.
    :returns: A dataset with complete metadata, five omics modalities, four
        drug views, pathway scores and the auxiliary ``uns`` payloads.
    """
    rng = np.random.default_rng(seed)
    cell_line_ids = _cell_line_ids()
    coverage = dict(omics_coverage or {})

    modalities: dict[str, ad.AnnData] = {"response": _response_anndata(rng, cell_line_ids)}
    for public_name, accessor in OMICS_ACCESSORS.items():
        n_covered = int(coverage.get(public_name, N_CELL_LINES))
        modalities[accessor] = _omics_anndata(public_name, rng, cell_line_ids, n_covered)

    md.set_options(pull_on_update=False)
    mdata = md.MuData(modalities)
    mdata.obs = modalities["response"].obs.copy()

    genes = synthetic_gene_symbols()
    mdata.uns["pathways_gmt"] = _pathways_gmt(genes)
    mdata.uns["sparsego"] = _sparsego_uns(genes)
    mdata.uns["bpe_codes"] = _bpe_codes([drug[2] for drug in _DRUGS])

    dataset = Dataset(mdata, name=name)
    _derived_drug_views(dataset)
    _pretrained_views(dataset, rng)
    return dataset
