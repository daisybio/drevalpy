"""Pytest configuration file for the tests directory."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from drevalpy.types.data.batch.response_batch import ResponseBatch


class MockFeatureSource:
    """Test helper satisfying the FeatureSource protocol."""

    def __init__(self, features: dict[str, dict[str, Any]], meta_info: dict[str, Any] | None = None):
        """Initialize with features dict and optional metadata.

        :param features: Mapping of entity_id -> {view_name -> feature_array}.
        :param meta_info: Optional mapping of view_name -> feature names or metadata.
        """
        self._features = features
        self._meta_info = meta_info or {}

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return np.array(list(self._features.keys()))

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """Direct access to the backing features dict (for legacy test code)."""
        return self._features

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return (len(ids), n_features) float array for a dense numeric view."""
        rows = [np.asarray(self._features[str(eid)][view], dtype=np.float64).ravel() for eid in entity_ids]
        return np.vstack(rows)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return ordered feature/column names for a view, or None."""
        meta = self._meta_info.get(view)
        return tuple(str(n) for n in meta) if meta is not None else None

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return the raw per-entity object for non-numeric views (graphs, etc.)."""
        entity = self._features.get(str(entity_id))
        if entity is None:
            return None
        return entity.get(view)

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata (e.g. ontology structures)."""
        return self._meta_info.get(key)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest session defaults and a headless Matplotlib backend.

    :param config: Pytest configuration object.
    """
    import matplotlib

    matplotlib.use("Agg")
    config.option.flaky_report = "none"
    config.option.tbstyle = "short"


_BUILTIN_MEASURE = "LN_IC50_curvecurator"

_TESTS_DIR = pathlib.Path(__file__).parent.resolve()
_DATA_DIR = (_TESTS_DIR.parent / "data").resolve()
_TOY_DATASETS = ("TOYv1", "TOYv2")

# Built-in dataset resolution now always goes through ``get_default_data_dir()``. Point it at
# the repo-local ``data/`` directory so existing test fixtures/downloads are found without
# every test having to pass a path explicitly. Set at import time (not as a fixture) so it is
# guaranteed to be in place before any other fixture or test module runs.
os.environ["DREVALPY_CACHE_DIR"] = str(_DATA_DIR)


def _load_toy_datasets() -> bool:
    """Check that TOYv1/TOYv2 CSV files exist for session fixtures.

    :returns: False when datasets are not available
    """
    try:
        for name in _TOY_DATASETS:
            csv_path = _DATA_DIR / name / f"{name}.csv"
            if not csv_path.is_file():
                import sys

                sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
                from download import download_dataset

                download_dataset(name, redownload=True)
    except Exception as exc:
        print(f"Warning: could not load TOY datasets: {exc}")
        return False
    return True


def _write_synthetic_smilesvec(
    smiles_file: pathlib.Path, output_file: pathlib.Path, *, embedding_dim: int = 100
) -> None:
    smiles_df = pd.read_csv(smiles_file, dtype=str)
    pubchem_ids = smiles_df["pubchem_id"].astype(str).tolist()
    rng = np.random.default_rng(seed=42)
    embeddings = rng.standard_normal((len(pubchem_ids), embedding_dim)).astype(np.float32)
    out_df = pd.DataFrame(embeddings, index=pubchem_ids)
    out_df.index.name = "pubchem_id"
    out_df.to_csv(output_file)


def _write_synthetic_bpe_smiles(smiles_file: pathlib.Path, output_file: pathlib.Path, *, max_length: int = 128) -> None:
    smiles_df = pd.read_csv(smiles_file, dtype=str)
    pubchem_ids = smiles_df["pubchem_id"].astype(str).tolist()
    rng = np.random.default_rng(seed=43)
    columns = [f"feature_{index}" for index in range(max_length)]
    data = rng.standard_normal((len(pubchem_ids), max_length)).astype(np.float32)
    out_df = pd.DataFrame(data, columns=columns)
    out_df.insert(0, "pubchem_id", pubchem_ids)
    out_df.to_csv(output_file, index=False)


def _ensure_bpe_smiles_features(path_data: str, dataset_name: str) -> None:
    dataset_dir = pathlib.Path(path_data) / dataset_name
    bpe_smiles_file = dataset_dir / "drug_bpe_smiles.csv"
    smiles_file = dataset_dir / "drug_smiles.csv"
    if bpe_smiles_file.exists():
        return
    if not smiles_file.exists():
        print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping BPE creation")
        return

    print(f"Creating synthetic BPE SMILES features for {dataset_name}...")
    _write_synthetic_bpe_smiles(smiles_file, bpe_smiles_file)


def _ensure_smilesvec_features(path_data: str, dataset_name: str) -> None:
    dataset_dir = pathlib.Path(path_data) / dataset_name
    smilesvec_file = dataset_dir / "drug_smilesvec.csv"
    smiles_file = dataset_dir / "drug_smiles.csv"
    if smilesvec_file.exists():
        return
    if not smiles_file.exists():
        print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping")
        return

    try:
        print(f"Creating synthetic SMILESVec drug features for {dataset_name}...")
        _write_synthetic_smilesvec(smiles_file, smilesvec_file)
    except Exception as exc:
        print(f"Warning: could not create drug features for {dataset_name}: {exc}")


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    """Fixture to provide the path to the data directory for tests.

    :returns: path to the data directory
    """
    return _DATA_DIR


@pytest.hookimpl(tryfirst=True)
@pytest.fixture(scope="session")
def sample_dataset(data_dir) -> ResponseBatch:
    """Sample dataset for testing individual models.

    :param data_dir: path to the data directory
    :returns: ResponseBatch with TOYv1 data
    """
    _ = data_dir
    csv_path = _DATA_DIR / "TOYv1" / "TOYv1.csv"
    df = pd.read_csv(csv_path)
    df = df[df[_BUILTIN_MEASURE].notna()]
    return ResponseBatch(
        response=df[_BUILTIN_MEASURE].to_numpy(dtype=np.float64),
        cell_line_ids=df["cellosaurus_id"].to_numpy(dtype=str),
        drug_ids=df["pubchem_id"].to_numpy(dtype=str),
    )


@pytest.fixture(scope="session")
def cross_study_dataset(data_dir) -> ResponseBatch:
    """Sample cross-study dataset for testing.

    :param data_dir: path to the data directory
    :returns: ResponseBatch with TOYv2 data
    """
    _ = data_dir
    csv_path = _DATA_DIR / "TOYv2" / "TOYv2.csv"
    df = pd.read_csv(csv_path)
    df = df[df[_BUILTIN_MEASURE].notna()]
    return ResponseBatch(
        response=df[_BUILTIN_MEASURE].to_numpy(dtype=np.float64),
        cell_line_ids=df["cellosaurus_id"].to_numpy(dtype=str),
        drug_ids=df["pubchem_id"].to_numpy(dtype=str),
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_bpe_features(data_dir) -> None:
    """Ensure BPE SMILES features are created for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that PharmaFormer
    and other models requiring BPE features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    if not _load_toy_datasets():
        return

    for dataset_name in _TOY_DATASETS:
        _ensure_bpe_smiles_features(path_data, dataset_name)


def _precily_gene_symbols_from_expr(expr_file: pathlib.Path) -> list[str] | None:
    if not expr_file.exists():
        return None
    with open(expr_file, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    non_gene_cols = {"cellosaurus_id", "cell_line_name"}
    return [column for column in header if column not in non_gene_cols]


def _write_synthetic_pathway_gmt(gmt_path: pathlib.Path, genes: list[str], min_size: int) -> None:
    gene_sets = {
        "SYNTH_PATHWAY_A": genes[: max(min_size, len(genes) // 2)],
        "SYNTH_PATHWAY_B": genes[-max(min_size, len(genes) // 2) :],
    }
    with open(gmt_path, "w", encoding="utf-8") as handle:
        for name, set_genes in gene_sets.items():
            handle.write("\t".join([name, "synthetic", *set_genes]) + "\n")


def _create_precily_pathway_features_for_dataset(
    path_data: str,
    dataset_name: str,
    create_precily_pathway_features,
    *,
    min_size: int = 5,
) -> None:
    dataset_dir = pathlib.Path(path_data) / dataset_name
    pathway_file = dataset_dir / "pathway_features.csv"
    if pathway_file.exists():
        return

    expr_file = dataset_dir / "gene_expression.csv"
    genes = _precily_gene_symbols_from_expr(expr_file)
    if genes is None:
        print(f"Warning: gene_expression.csv not found for {dataset_name}, skipping")
        return
    if len(genes) < min_size:
        print(f"Warning: too few genes in {dataset_name} ({len(genes)}), skipping")
        return

    gmt_path = dataset_dir / "synthetic_pathways.gmt"
    _write_synthetic_pathway_gmt(gmt_path, genes, min_size)
    try:
        print(f"Creating synthetic GSVA pathway features for {dataset_name}...")
        create_precily_pathway_features(
            data_path=path_data,
            dataset_name=dataset_name,
            gene_sets=str(gmt_path),
            min_size=min_size,
        )
    except Exception as exc:
        print(f"Warning: could not create pathway features for {dataset_name}: {exc}")
        import traceback

        traceback.print_exc()


@pytest.fixture(scope="session", autouse=True)
def ensure_precily_pathway_features(data_dir) -> None:
    """Ensure GSVA pathway features exist for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that Precily
    and other models requiring Precily features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)

    try:
        from gseapy import gsva  # noqa: F401

        _has_gseapy = True
    except ImportError:
        _has_gseapy = False

    if not _has_gseapy:
        return

    def _csv_create_precily_pathway_features(data_path, dataset_name, gene_sets, min_size=5, max_size=2000):
        """Lightweight CSV-based pathway feature generation for tests."""
        import gseapy as gp

        data_root = pathlib.Path(data_path)
        expr_file = data_root / dataset_name / "gene_expression.csv"
        import pandas as pd

        expr = pd.read_csv(expr_file, index_col=0).select_dtypes(include="number")
        expr = expr.loc[~expr.index.duplicated(keep="first")]
        gv = gp.gsva(
            data=expr.T,
            gene_sets=gene_sets,
            kcdf="Gaussian",
            min_size=min_size,
            max_size=max_size,
            mx_diff=True,
            threads=1,
            seed=42,
            outdir=None,
            verbose=False,
        )
        long = gv.res2d.copy()
        cols_map = {c.lower(): c for c in long.columns}
        term_col = cols_map.get("term", "Term")
        name_col = cols_map.get("name", "Name")
        es_col = cols_map.get("es", cols_map.get("nes", "ES"))
        import numpy as np

        wide = long.pivot(index=term_col, columns=name_col, values=es_col)
        scores = wide.T.astype(np.float32)
        scores.index.name = "cell_line_name"
        out_path = data_root / dataset_name / "pathway_features.csv"
        scores.to_csv(out_path)

    # Ensure datasets are loaded first (this will download them if needed)
    if not _load_toy_datasets():
        return

    for dataset_name in _TOY_DATASETS:
        _create_precily_pathway_features_for_dataset(
            path_data,
            dataset_name,
            _csv_create_precily_pathway_features,
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_precily_drug_features(data_dir) -> None:
    """Ensure SMILESVec drug features exist for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that Precily
    and other models requiring Precily features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    if not _load_toy_datasets():
        return

    for dataset_name in _TOY_DATASETS:
        _ensure_smilesvec_features(path_data, dataset_name)


@pytest.fixture(scope="session", autouse=True)
def ensure_sparsego_ontology_features(data_dir) -> None:
    """Ensure SparseGO ontology features exist for TOYv1 and TOYv2 before tests run.

    Prefers committed fixtures under ``tests/fixtures/sparsego/`` so CI does not
    depend on MyGene.info / GO network calls. Falls back to generating from
    go-basic.obo when fixtures are absent and optional deps are installed.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    fixture_root = _TESTS_DIR / "fixtures" / "sparsego"

    # Ensure datasets are loaded first (this will download them if needed)
    if not _load_toy_datasets():
        return

    for dataset_name in _TOY_DATASETS:
        dataset_dir = pathlib.Path(path_data) / dataset_name
        ont_file = dataset_dir / "sparseGO_ont.txt"
        gene2ind_file = dataset_dir / "gene2ind.txt"
        if ont_file.exists() and gene2ind_file.exists():
            continue

        fixture_dir = fixture_root / dataset_name
        fixture_ont = fixture_dir / "sparseGO_ont.txt"
        fixture_gene2ind = fixture_dir / "gene2ind.txt"
        if fixture_ont.exists() and fixture_gene2ind.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
            ont_file.write_bytes(fixture_ont.read_bytes())
            gene2ind_file.write_bytes(fixture_gene2ind.read_bytes())
            continue

        expr_file = dataset_dir / "gene_expression.csv"
        if not expr_file.exists():
            print(f"Warning: gene_expression.csv not found for {dataset_name}, skipping")
            continue

        try:
            import mygene  # noqa: F401
            import obonet  # noqa: F401
        except ImportError:
            print(f"Warning: SparseGO dependencies (mygene, obonet) unavailable for {dataset_name}")
            continue

        try:
            import sys

            _scripts_dir = str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "featurizer")
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            import pandas as pd
            from sparsego_graph import build_pruned_graph, fetch_gene_go_annotations

            expr_genes = pd.read_csv(expr_file, index_col=0, nrows=0).columns.tolist()
            gene_go_df = fetch_gene_go_annotations(expr_genes)
            our_graph = build_pruned_graph(gene_go_df, None, n=5, m=10, p=8)

            import numpy as np

            edges = np.array(list(our_graph.edges()))
            edges = np.unique(edges, axis=0)
            type_col = np.where(np.char.startswith(edges[:, 1].astype(str), "GO:"), "default", "gene")
            edges_with_type = np.column_stack([edges, type_col])
            pd.DataFrame(edges_with_type).to_csv(ont_file, sep="\t", index=False, header=False)

            gene_edges = edges_with_type[edges_with_type[:, 2] == "gene"]
            keep_genes = sorted(set(gene_edges[:, 1]))
            with open(gene2ind_file, "w") as fh:
                for idx, gene in enumerate(keep_genes):
                    fh.write(f"{idx}\t{gene}\n")
            print(f"SparseGO ontology features created for {dataset_name}")
        except Exception as e:
            print(f"Warning: could not create SparseGO ontology features for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()


@pytest.fixture(scope="session", autouse=True)
def ensure_model_drug_embeddings(
    data_dir,
    sample_dataset,
    cross_study_dataset,
    ensure_sparsego_ontology_features,
) -> None:
    """Re-ensure PharmaFormer/Precily drug embeddings after TOY datasets are loaded.

    Earlier autouse fixtures may run before the first successful TOY download on CI;
    this pass creates any still-missing embedding files once ``sample_dataset`` is ready.

    :param data_dir: path to the data directory
    :param sample_dataset: ensures TOYv1 is present
    :param cross_study_dataset: ensures TOYv2 is present
    :param ensure_sparsego_ontology_features: run after other feature fixtures
    """
    _ = sample_dataset, cross_study_dataset, ensure_sparsego_ontology_features
    path_data = str(data_dir)
    for dataset_name in _TOY_DATASETS:
        _ensure_bpe_smiles_features(path_data, dataset_name)
        _ensure_smilesvec_features(path_data, dataset_name)
