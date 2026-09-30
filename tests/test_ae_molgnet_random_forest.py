"""Tests for RandomForestAEMolGNet, the Random Forest on DIPK's feature building blocks."""

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.baselines import ae_molgnet_random_forest as module
from drevalpy.models.baselines.ae_molgnet_random_forest import (
    MOLGNET_POOLED_VIEW,
    RandomForestAEMolGNet,
    molgnet_feature_names,
    pool_molgnet,
)

#: Eight cell lines, so that the autoencoder gets a training matrix of more than one batch.
_CELL_LINES = ["CAL-120", "DMS 114", "CAL-51", "H2126", "HCC1143", "MCF7", "A549", "K562"]
_ALL_GENES = ["TSPAN6", "TNMD", "BRCA1", "SCYL3", "HDAC1", "INSIG1", "FOXO3", "MYC"]
_SUBSET_GENES = ["BRCA1", "SCYL3", "HDAC1", "MYC"]
_DRUGS = ["1234", "5678"]

#: Atom counts differ per drug on purpose: pooling has to even them out to one length.
_ATOMS = {"1234": 5, "5678": 9}
_MOLGNET_DIM = 6

#: Hyperparameters that keep the autoencoder cheap enough for a unit test.
_HPAMS = {
    "cell_line_views": ["gene_expression"],
    "drug_views": [MOLGNET_POOLED_VIEW],
    "epochs_autoencoder": 1,
    "patience_autoencoder": 1,
    "batch_size_autoencoder": 4,
    "n_estimators": 3,
    "max_depth": 2,
    "max_samples": None,
    "criterion": "squared_error",
    "n_jobs": 1,
}


def _gene_expression_csv(genes: list[str]) -> str:
    """Build a gene expression CSV on a linear scale.

    :param genes: gene symbols to write as columns
    :returns: CSV text
    """
    header = "cellosaurus_id,cell_line_name," + ",".join(genes) + "\n"
    rows = []
    for index, cell_line in enumerate(_CELL_LINES):
        values = [f"{(index + 1) * (position + 1) * 1.5:.2f}" for position in range(len(genes))]
        rows.append(f"CVCL_{1000 + index},{cell_line}," + ",".join(values) + "\n")
    return header + "".join(rows)


@pytest.fixture
def molgnet_data_dir(tmp_path) -> str:
    """Build a minimal data directory with gene expression, a gene list and MolGNet files.

    :param tmp_path: pytest temporary path fixture
    :returns: path to the data directory
    """
    dataset_dir = tmp_path / "TOY_small"
    dataset_dir.mkdir()
    (dataset_dir / "gene_expression.csv").write_text(_gene_expression_csv(_ALL_GENES), encoding="utf-8")

    drugs_dir = dataset_dir / "DIPK_features" / "Drugs"
    drugs_dir.mkdir(parents=True)
    for drug, n_atoms in _ATOMS.items():
        header = "\t" + "\t".join(f"dim_{i}" for i in range(_MOLGNET_DIM)) + "\n"
        rows = ""
        for atom in range(n_atoms):
            values = [f"{(atom + 1) * (dim + 1) * 0.25:.4f}" for dim in range(_MOLGNET_DIM)]
            rows += f"atom_{atom}\t" + "\t".join(values) + "\n"
        (drugs_dir / f"MolGNet_{drug}.csv").write_text(header + rows, encoding="utf-8")

    gene_lists = tmp_path / "meta" / "gene_lists"
    gene_lists.mkdir(parents=True)
    (gene_lists / "gene_expression_intersection.csv").write_text(
        "Symbol\n" + "\n".join(_ALL_GENES) + "\n", encoding="utf-8"
    )
    (gene_lists / "subset_genes.csv").write_text("Symbol\n" + "\n".join(_SUBSET_GENES) + "\n", encoding="utf-8")
    return str(tmp_path)


def _build(**overrides) -> RandomForestAEMolGNet:
    """Build a model with the cheap test hyperparameters.

    :param overrides: hyperparameters overriding the defaults
    :returns: built model
    """
    model = RandomForestAEMolGNet()
    model.build_model({**_HPAMS, **overrides})
    return model


def _response() -> DrugResponseDataset:
    """Build a response dataset covering every cell line and drug.

    :returns: response dataset
    """
    cell_line_ids, drug_ids, response = [], [], []
    for index, cell_line in enumerate(_CELL_LINES):
        for position, drug in enumerate(_DRUGS):
            cell_line_ids.append(cell_line)
            drug_ids.append(drug)
            response.append(float(index) * 0.3 + float(position))
    return DrugResponseDataset(
        response=np.array(response),
        cell_line_ids=np.array(cell_line_ids),
        drug_ids=np.array(drug_ids),
    )


# ------------------------------------------------------------------ pooling


def test_pool_molgnet_mean_and_max() -> None:
    """Mean and max pooling must reduce the atom axis and keep the embedding dimension."""
    nodes = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    assert pool_molgnet(nodes, "mean").tolist() == [2.0, 3.0]
    assert pool_molgnet(nodes, "max").tolist() == [3.0, 4.0]


def test_pool_molgnet_mean_max_concatenates() -> None:
    """mean_max has to concatenate both, doubling the length."""
    nodes = np.array([[1.0, 4.0], [3.0, 2.0]], dtype=np.float32)
    assert pool_molgnet(nodes, "mean_max").tolist() == [2.0, 3.0, 3.0, 4.0]


def test_pool_molgnet_rejects_unknown_pooling() -> None:
    """An unknown pooling name must fail loudly instead of silently picking a default."""
    with pytest.raises(ValueError, match="Unknown MolGNet pooling"):
        pool_molgnet(np.zeros((2, 2), dtype=np.float32), "median")


def test_molgnet_feature_names_distinguish_mean_and_max() -> None:
    """The names of a mean_max view must say which half a feature comes from."""
    names = molgnet_feature_names(4, "mean_max")
    assert names == ["molgnet_mean_0", "molgnet_mean_1", "molgnet_max_0", "molgnet_max_1"]
    assert molgnet_feature_names(2, "mean") == ["molgnet_mean_0", "molgnet_mean_1"]


# ------------------------------------------------------------------ drug features


def test_load_drug_features_pools_molecules_of_different_size(molgnet_data_dir) -> None:
    """Molecules with different atom counts must end up with the same feature length.

    :param molgnet_data_dir: path to the temporary data directory
    """
    features = _build().load_drug_features(molgnet_data_dir, "TOY_small")
    assert sorted(features.features.keys()) == sorted(_DRUGS)
    lengths = {len(features.features[drug][MOLGNET_POOLED_VIEW]) for drug in _DRUGS}
    assert lengths == {2 * _MOLGNET_DIM}


def test_load_drug_features_honours_pooling_hyperparameter(molgnet_data_dir) -> None:
    """Single-statistic pooling must halve the number of features.

    :param molgnet_data_dir: path to the temporary data directory
    """
    features = _build(molgnet_pooling="mean").load_drug_features(molgnet_data_dir, "TOY_small")
    assert len(features.features["1234"][MOLGNET_POOLED_VIEW]) == _MOLGNET_DIM
    assert list(features.meta_info[MOLGNET_POOLED_VIEW])[0] == "molgnet_mean_0"


def test_load_drug_features_reads_tab_separated_values(molgnet_data_dir) -> None:
    """The MolGNet files are tab separated; reading them as commas would give one column.

    :param molgnet_data_dir: path to the temporary data directory
    """
    features = _build(molgnet_pooling="max").load_drug_features(molgnet_data_dir, "TOY_small")
    expected = [(_ATOMS["1234"]) * (dim + 1) * 0.25 for dim in range(_MOLGNET_DIM)]
    np.testing.assert_allclose(features.features["1234"][MOLGNET_POOLED_VIEW], expected, rtol=1e-6)


def test_load_drug_features_without_dipk_files(tmp_path) -> None:
    """A dataset without DIPK features must raise, not return an empty FeatureDataset.

    :param tmp_path: pytest temporary path fixture
    """
    (tmp_path / "TOY_small").mkdir()
    with pytest.raises(FileNotFoundError, match="MolGNet features not found"):
        _build().load_drug_features(str(tmp_path), "TOY_small")


# ------------------------------------------------------------------ gene expression handling


def test_log2_transform_is_applied(molgnet_data_dir) -> None:
    """Gene expression must reach the autoencoder on a log2 scale by default.

    :param molgnet_data_dir: path to the temporary data directory
    """
    raw = _build(autoencoder_log2_transform=False).load_cell_line_features(molgnet_data_dir, "TOY_small")
    logged = _build().load_cell_line_features(molgnet_data_dir, "TOY_small")
    raw_values = raw.features["CAL-120"]["gene_expression"]
    log_values = logged.features["CAL-120"]["gene_expression"]
    np.testing.assert_allclose(log_values, np.log2(raw_values + 1.0), rtol=1e-6)


def test_gene_list_subsets_the_autoencoder_input(molgnet_data_dir) -> None:
    """The gene_list hyperparameter has to restrict the autoencoder input space.

    :param molgnet_data_dir: path to the temporary data directory
    """
    features = _build(gene_list="subset_genes").load_cell_line_features(molgnet_data_dir, "TOY_small")
    assert sorted(features.meta_info["gene_expression"]) == sorted(_SUBSET_GENES)


def test_default_gene_list_matches_dipk() -> None:
    """Both models must encode the same gene space unless told otherwise."""
    from drevalpy.models.DIPK.dipk import DIPKModel

    assert RandomForestAEMolGNet.gene_list == DIPKModel.gene_list


# ------------------------------------------------------------------ autoencoder behaviour


def test_autoencoder_never_sees_test_cell_lines(molgnet_data_dir, monkeypatch) -> None:
    """The encoder may only be fitted on training cell lines, otherwise the split leaks.

    :param molgnet_data_dir: path to the temporary data directory
    :param monkeypatch: pytest monkeypatch fixture
    """
    seen: dict[str, np.ndarray] = {}
    original = module.train_gene_expession_autoencoder

    def spy(train_matrix, val_matrix, **kwargs):
        seen["n_rows"] = len(train_matrix) + len(val_matrix)
        return original(train_matrix, val_matrix, **kwargs)

    monkeypatch.setattr(module, "train_gene_expession_autoencoder", spy)

    model = _build()
    cell_line_input = model.load_cell_line_features(molgnet_data_dir, "TOY_small")
    drug_input = model.load_drug_features(molgnet_data_dir, "TOY_small")
    train_lines = _CELL_LINES[:6]
    output = _response()
    keep = np.isin(output.cell_line_ids, train_lines)
    training = DrugResponseDataset(
        response=output.response[keep],
        cell_line_ids=output.cell_line_ids[keep],
        drug_ids=output.drug_ids[keep],
    )
    model.train(output=training, cell_line_input=cell_line_input, drug_input=drug_input)

    assert seen["n_rows"] == len(train_lines)


def test_autoencoder_trains_on_unique_cell_lines_not_response_rows(molgnet_data_dir, monkeypatch) -> None:
    """One row per cell line, not one per response, is the documented difference to DIPK.

    :param molgnet_data_dir: path to the temporary data directory
    :param monkeypatch: pytest monkeypatch fixture
    """
    seen: dict[str, int] = {}
    original = module.train_gene_expession_autoencoder

    def spy(train_matrix, val_matrix, **kwargs):
        seen["n_rows"] = len(train_matrix) + len(val_matrix)
        return original(train_matrix, val_matrix, **kwargs)

    monkeypatch.setattr(module, "train_gene_expession_autoencoder", spy)
    model = _build()
    output = _response()
    model.train(
        output=output,
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    assert len(output) == len(_CELL_LINES) * len(_DRUGS)
    assert seen["n_rows"] == len(_CELL_LINES)


def test_batchnorm_guard_avoids_a_trailing_single_row(molgnet_data_dir, monkeypatch) -> None:
    """A training matrix of n % batch_size == 1 must not reach BatchNorm1d as a batch of one.

    :param molgnet_data_dir: path to the temporary data directory
    :param monkeypatch: pytest monkeypatch fixture
    """
    seen: dict[str, int] = {}
    original = module.train_gene_expession_autoencoder

    def spy(train_matrix, val_matrix, **kwargs):
        seen["n_train"] = len(train_matrix)
        return original(train_matrix, val_matrix, **kwargs)

    monkeypatch.setattr(module, "train_gene_expession_autoencoder", spy)
    # 8 cell lines, one of them validation, batch size 3 -> 7 training rows, 7 % 3 == 1.
    model = _build(batch_size_autoencoder=3, autoencoder_validation_fraction=0.1)
    model.train(
        output=_response(),
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    assert seen["n_train"] % 3 != 1


def test_encoding_reduces_features_to_the_latent_dimension(molgnet_data_dir) -> None:
    """After training, the gene expression view must hold latent vectors, not genes.

    :param molgnet_data_dir: path to the temporary data directory
    """
    model = _build()
    cell_line_input = model.load_cell_line_features(molgnet_data_dir, "TOY_small")
    model.train(
        output=_response(),
        cell_line_input=cell_line_input,
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    encoded = model._encode(model.load_cell_line_features(molgnet_data_dir, "TOY_small"))
    assert model._encoder is not None
    assert len(encoded.features["CAL-120"]["gene_expression"]) == model._encoder.latent_dim


def test_encoder_rejects_a_different_gene_space(molgnet_data_dir) -> None:
    """An encoder must refuse features from a gene space it was not trained on.

    The encoder is only ever valid for the genes it saw. Feeding it a different number of genes --
    a different ``gene_list``, or a pretrained encoder from another gene space -- has to fail with
    a named error instead of a matrix shape error, and must never silently return numbers.

    :param molgnet_data_dir: path to the temporary data directory
    """
    model = _build()
    cell_line_input = model.load_cell_line_features(molgnet_data_dir, "TOY_small")
    model.train(
        output=_response(),
        cell_line_input=cell_line_input,
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    assert model._ae_input_dim == len(_ALL_GENES)

    narrower = model.load_cell_line_features(molgnet_data_dir, "TOY_small")
    for cell_line in narrower.features:
        narrower.features[cell_line]["gene_expression"] = narrower.features[cell_line]["gene_expression"][:-1]
    with pytest.raises(ValueError, match="was built for"):
        model._encode(narrower)


# ------------------------------------------------------------------ encoder cache


def test_encoder_cache_is_written_and_reused(molgnet_data_dir, tmp_path, monkeypatch) -> None:
    """With a cache directory the encoder must be trained once and loaded afterwards.

    :param molgnet_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    calls = {"n": 0}
    original = module.train_gene_expession_autoencoder

    def spy(train_matrix, val_matrix, **kwargs):
        calls["n"] += 1
        return original(train_matrix, val_matrix, **kwargs)

    monkeypatch.setattr(module, "train_gene_expession_autoencoder", spy)
    cache = tmp_path / "ae_cache"

    for _ in range(2):
        model = _build(autoencoder_cache_dir=str(cache))
        model.train(
            output=_response(),
            cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
            drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
        )

    assert calls["n"] == 1
    assert len(list(cache.glob("gene_ae_*.pt"))) == 1


def test_cache_is_disabled_without_a_directory(molgnet_data_dir, monkeypatch) -> None:
    """Without a configured directory nothing may be cached, not even to a default location.

    :param molgnet_data_dir: path to the temporary data directory
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.delenv("DREVAL_AE_CACHE", raising=False)
    model = _build()
    assert model._cache_path(np.array(_CELL_LINES), len(_ALL_GENES), _ALL_GENES) is None


def test_cache_key_depends_on_gene_order(molgnet_data_dir, tmp_path) -> None:
    """Two runs differing only in gene order must not share a cached encoder.

    :param molgnet_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    """
    model = _build(autoencoder_cache_dir=str(tmp_path / "cache"))
    ids = np.array(_CELL_LINES)
    forward = model._cache_path(ids, len(_ALL_GENES), _ALL_GENES)
    reversed_order = model._cache_path(ids, len(_ALL_GENES), list(reversed(_ALL_GENES)))
    assert forward is not None and forward != reversed_order


def test_cache_key_depends_on_training_cell_lines(molgnet_data_dir, tmp_path) -> None:
    """A different fold must get its own encoder.

    :param molgnet_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    """
    model = _build(autoencoder_cache_dir=str(tmp_path / "cache"))
    full = model._cache_path(np.array(_CELL_LINES), len(_ALL_GENES), _ALL_GENES)
    fold = model._cache_path(np.array(_CELL_LINES[:6]), len(_ALL_GENES), _ALL_GENES)
    assert full != fold


def test_environment_variable_configures_the_cache(molgnet_data_dir, tmp_path, monkeypatch) -> None:
    """DREVAL_AE_CACHE must work as a fallback for the hyperparameter.

    :param molgnet_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setenv("DREVAL_AE_CACHE", str(tmp_path / "from_env"))
    model = _build()
    path = model._cache_path(np.array(_CELL_LINES), len(_ALL_GENES), _ALL_GENES)
    assert path is not None and str(tmp_path / "from_env") in path


# ------------------------------------------------------------------ persistence and integration


def test_save_and_load_restore_identical_predictions(molgnet_data_dir, tmp_path) -> None:
    """Without the encoder a reloaded model would silently predict on raw genes.

    :param molgnet_data_dir: path to the temporary data directory
    :param tmp_path: pytest temporary path fixture
    """
    model = _build()
    output = _response()
    model.train(
        output=output,
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    before = model.predict(
        cell_line_ids=output.cell_line_ids,
        drug_ids=output.drug_ids,
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )

    directory = tmp_path / "saved_model"
    model.save(str(directory))
    assert (directory / "gene_encoder.pt").exists()

    restored = RandomForestAEMolGNet.load(str(directory))
    after = restored.predict(
        cell_line_ids=output.cell_line_ids,
        drug_ids=output.drug_ids,
        cell_line_input=restored.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=restored.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    np.testing.assert_allclose(before, after, rtol=1e-5)


def test_train_and_predict_end_to_end(molgnet_data_dir) -> None:
    """A full train and predict cycle must return one finite prediction per response row.

    :param molgnet_data_dir: path to the temporary data directory
    """
    model = _build()
    output = _response()
    model.train(
        output=output,
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    predictions = model.predict(
        cell_line_ids=output.cell_line_ids,
        drug_ids=output.drug_ids,
        cell_line_input=model.load_cell_line_features(molgnet_data_dir, "TOY_small"),
        drug_input=model.load_drug_features(molgnet_data_dir, "TOY_small"),
    )
    assert predictions.shape == output.response.shape
    assert np.all(np.isfinite(predictions))


def test_registered_in_model_factory() -> None:
    """The model has to be reachable through the factory the pipeline uses."""
    assert MODEL_FACTORY["RandomForestAEMolGNet"] is RandomForestAEMolGNet
    assert RandomForestAEMolGNet.get_model_name() == "RandomForestAEMolGNet"


def test_hyperparameters_are_declared() -> None:
    """The shipped hyperparameters.yaml must cover the model, otherwise tuning cannot run."""
    grid = RandomForestAEMolGNet.get_hyperparameter_set()
    assert grid, "no hyperparameter grid found"
    first = grid[0]
    # ParameterGrid unpacks each yaml list into single values.
    assert first["drug_views"] == MOLGNET_POOLED_VIEW
    assert first["batch_size_autoencoder"] == 32
    assert first["gene_list"] == "gene_expression_intersection"
