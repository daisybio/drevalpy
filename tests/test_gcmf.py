"""Smoke tests for the GCMF model family (GCMF, PGCMF, RGCMF, PRGCMF)."""

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY

GCMF_FAMILY = ["GCMF", "PGCMF", "RGCMF", "PRGCMF"]


def test_gcmf_family_in_factory() -> None:
    """All four GCMF-family models are registered in the model factory."""
    for name in GCMF_FAMILY:
        assert name in MODEL_FACTORY


def test_default_relations_ship_with_meta_bundle(data_dir, sample_dataset) -> None:
    """
    Every resource the default RGCMF relations read is present in the downloaded meta bundle.

    Guards the data dependency directly, on both sides: the drug relations under
    ``meta/gcmf_drug_relations/`` and the per-omics gene lists the cell relations reduce with.
    If any is ever dropped from the bundle this fails here with a clear message instead of
    surfacing as a training error downstream.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: session fixture; ensures the toy data and meta bundle are downloaded
    """
    from drevalpy.models.GCMF.gcmf import RGCMF

    model = RGCMF()
    for view in model.drug_relation_views:
        assert RGCMF._drug_resource_path(view, data_path=str(data_dir)) is not None, (
            f"drug relation '{view}' is missing from <data>/meta/{RGCMF._DRUG_SIM_DIR}/; "
            "it must ship with the meta bundle"
        )
    # the gene_expression relation is built from the node features (hp gene_list), so only the
    # omics views loaded from disk need their gene list to be present
    for view in model.cell_relation_views:
        if view == "gene_expression":
            continue
        gene_list = RGCMF._CELL_GENE_LISTS[view]
        path = data_dir / "meta" / "gene_lists" / f"{gene_list}.csv"
        assert path.is_file(), (
            f"gene list '{gene_list}' for cell relation '{view}' is missing from "
            "<data>/meta/gene_lists/; it must ship with the meta bundle"
        )


def _tiny_hpams(model_cls) -> dict:
    """
    First hyperparameter set with the expensive knobs shrunk so a smoke test runs fast.

    :param model_cls: the GCMF-family model class to take the hyperparameter set from
    :returns: the shrunken hyperparameter dictionary
    """
    hp = dict(model_cls.get_hyperparameter_set()[0])
    hp.update(
        n_ensemble=1,
        max_epochs=2,
        patience=2,
        hidden_dim=32,
        emb_dim=16,
        mlp_hidden=16,
        batch_size=64,
        k_cell=4,
        k_drug=4,
        n_bits=128,  # toy data ships only 128-bit fingerprints
    )
    # The relational models keep their real relation defaults on both sides (four cell relations;
    # drug_pathways + drug_bioassay) and their real gene list, so the test exercises every
    # resource they read from the meta bundle and fails if any of them goes missing.
    # The base models default to landmark_genes, which the toy meta bundle does not ship, so only
    # they need the gene list relaxed to all available genes.
    if model_cls.get_model_name() in ("GCMF", "PGCMF"):
        hp["gene_list"] = None
    return hp


@pytest.mark.parametrize("model_name", GCMF_FAMILY)
def test_gcmf_family_train_predict_save_load(model_name: str, data_dir, sample_dataset) -> None:
    """
    Each model trains on TOYv1's measured responses, predicts, and round-trips save/load.

    The targets are the real TOYv1 responses rather than synthetic values, so the model is
    trained on actual signal and the fit is checked, not just the plumbing.

    :param model_name: name of the GCMF-family model under test
    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: measured TOYv1 responses (session fixture from conftest)
    """
    data_path = str(data_dir)
    model = MODEL_FACTORY[model_name]()
    model.build_model(_tiny_hpams(type(model)))
    cell_input = model.load_cell_line_features(data_path=data_path, dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=data_path, dataset_name="TOYv1")

    # copied off the session fixture so splitting it elsewhere cannot affect this test
    cl_ids = np.asarray(sample_dataset.cell_line_ids)
    dr_ids = np.asarray(sample_dataset.drug_ids)
    responses = np.asarray(sample_dataset.response, dtype=float)
    train = DrugResponseDataset(response=responses, cell_line_ids=cl_ids, drug_ids=dr_ids, dataset_name="TOYv1")

    model.train(output=train, cell_line_input=cell_input, drug_input=drug_input)
    preds = model.predict(cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input)
    assert preds.shape == cl_ids.shape
    assert np.isfinite(preds).all()
    # training on real responses has to leave the predictions correlated with them; a model that
    # learned nothing, or collapsed to a constant, would not clear this
    assert np.corrcoef(preds, responses)[0, 1] > 0.2

    # save -> load -> predict must reproduce the same predictions
    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        reloaded = type(model).load(directory)
    preds_reloaded = reloaded.predict(
        cell_line_ids=cl_ids, drug_ids=dr_ids, cell_line_input=cell_input, drug_input=drug_input
    )
    assert np.allclose(preds, preds_reloaded, atol=1e-4)


def test_kendall_contingency_matches_scipy() -> None:
    """
    The vectorized Kendall used for copy number reproduces ``scipy.stats.kendalltau`` exactly.

    The graphs are rebuilt on every train() rather than cached, which is only affordable because
    this path replaces the O(n^2) loop of scipy calls. It has to be the same tau-b, ties included,
    or the relation silently changes meaning.
    """
    from scipy.stats import kendalltau

    from drevalpy.models.GCMF.gcmf import _similarity_matrix

    rng = np.random.default_rng(0)
    x = rng.integers(-2, 3, size=(40, 60)).astype(float)  # GISTIC-like: five ordinal levels
    x[3, :] = 1.0  # a constant row exercises the zero-denominator branch
    fast = _similarity_matrix(x, "kendall")

    expected = np.ones((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            tau = kendalltau(x[i], x[j])[0]
            expected[i, j] = expected[j, i] = 0.0 if np.isnan(tau) else tau
    # _similarity_matrix maps tau from [-1, 1] to [0, 1], so compare against the mapped reference
    assert np.abs(fast - (expected + 1.0) / 2.0).max() < 1e-12


def test_rgcmf_cell_graphs_follow_randomized_features(data_dir, sample_dataset) -> None:
    """
    Randomizing a cell-line view rebuilds that relation's graph and leaves the others alone.

    This is what makes the SVCC/SVRC randomization tests meaningful for RGCMF: the graphs are the
    model's main inductive bias, so if they were derived once at load time they would survive the
    permutation intact and the model would look far more robust to randomized features than it is.

    :param data_dir: path to the test data directory (session fixture from conftest)
    :param sample_dataset: session fixture; ensures the toy data and meta bundle are downloaded
    """
    import torch

    from drevalpy.models.GCMF.gcmf import RGCMF

    model = RGCMF()
    model.build_model(_tiny_hpams(RGCMF))
    cell_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    cell_ids = np.unique(cell_input.identifiers)
    # every omics view the model builds a graph from must be randomizable by the framework
    assert model.cell_relation_views == model.cell_line_views
    x_cell = np.zeros((len(cell_ids), 1), dtype=np.float32)  # unused by the relational builder

    adj = model._build_cell_adj(x_cell, cell_input, cell_ids, model.hyperparameters)
    randomized = cell_input.copy()
    randomized.randomize_features("gene_expression", randomization_type="permutation")
    adj_randomized = model._build_cell_adj(x_cell, randomized, cell_ids, model.hyperparameters)

    gene_expression = model.cell_relation_views.index("gene_expression")
    assert not torch.allclose(adj[gene_expression], adj_randomized[gene_expression])
    for i, view in enumerate(model.cell_relation_views):
        if i != gene_expression:
            assert torch.allclose(adj[i], adj_randomized[i]), f"relation {view} changed unexpectedly"


def test_rgcmf_drug_relation_resource_resolution(tmp_path) -> None:
    """A drug relation resolves from ``<data_path>/meta/<dir>/`` and is None when absent.

    :param tmp_path: pytest-provided temporary directory
    """
    from drevalpy.models.GCMF.gcmf import RGCMF

    assert RGCMF._drug_resource_path("drug_pathways", data_path=str(tmp_path)) is None

    meta = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    meta.mkdir(parents=True)
    (meta / "drug_pathways.csv.gz").write_bytes(b"")  # presence is enough for path resolution
    resolved = RGCMF._drug_resource_path("drug_pathways", data_path=str(tmp_path))
    assert resolved is not None and resolved.endswith("drug_pathways.csv.gz")


def test_rgcmf_raises_on_relation_without_pubchem_id(tmp_path, data_dir) -> None:
    """A relation table lacking the pubchem_id join key is rejected instead of guessed at.

    :param tmp_path: pytest-provided temporary directory
    :param data_dir: path to the test data directory (source of the toy dataset)
    """
    import shutil

    import pandas as pd

    from drevalpy.models.GCMF.gcmf import RGCMF

    shutil.copytree(data_dir / "TOYv1", tmp_path / "TOYv1")
    shutil.copytree(data_dir / "meta", tmp_path / "meta")
    rel_dir = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    for stale in rel_dir.glob("drug_pathways.csv*"):
        stale.unlink()
    # an old-style table keyed on drug name; the join key is checked before any row is read,
    # so the header alone is what this exercises
    pd.DataFrame(columns=["drug_name", "pathway"]).to_csv(rel_dir / "drug_pathways.csv", index=False)

    model = RGCMF()
    model.build_model(_tiny_hpams(RGCMF))
    with pytest.raises(ValueError, match="pubchem_id"):
        model.load_drug_features(data_path=str(tmp_path), dataset_name="TOYv1")


def test_rgcmf_raises_on_missing_drug_relation(tmp_path, data_dir) -> None:
    """RGCMF raises (no silent fallback) when a configured drug-relation resource is absent.

    :param tmp_path: pytest-provided temporary directory (its meta has no relation resource)
    :param data_dir: path to the test data directory (source of the toy dataset)
    """
    import shutil

    from drevalpy.models.GCMF.gcmf import RGCMF

    shutil.copytree(data_dir / "TOYv1", tmp_path / "TOYv1")
    shutil.copytree(data_dir / "meta", tmp_path / "meta")
    rel_dir = tmp_path / "meta" / RGCMF._DRUG_SIM_DIR
    if rel_dir.exists():
        shutil.rmtree(rel_dir)

    model = RGCMF()
    model.build_model(_tiny_hpams(RGCMF))
    model.load_cell_line_features(data_path=str(tmp_path), dataset_name="TOYv1")
    with pytest.raises(FileNotFoundError):
        model.load_drug_features(data_path=str(tmp_path), dataset_name="TOYv1")
