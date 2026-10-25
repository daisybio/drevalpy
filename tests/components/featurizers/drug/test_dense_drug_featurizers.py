"""Tests for the dense single-view drug featurizers: view, fingerprints, ChemBERTa and SMILESVec.

Only the real weight downloads are ``network``-marked; the loaders are exercised offline with tiny stand-ins.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.drug import chemberta, smilesvec
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.drug.chemberta import ChemBertaFeaturizer, load_chemberta
from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer, _fingerprint_for_smiles
from drevalpy.components.featurizers.drug.smilesvec import SmilesVecDrugFeaturizer, _smilesvec_embed
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.data import artifacts
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.components.featurizers._helpers import SmilesSource
from tests.conftest import MockFeatureSource

_EMBEDDERS = [
    pytest.param(FingerprintsFeaturizer, "morgan_fingerprint", "fingerprints", ("fp1", "fp2"), id="fingerprints"),
    pytest.param(ChemBertaFeaturizer, "chemberta", "chemberta", ("e1", "e2"), id="chemberta"),
    pytest.param(SmilesVecDrugFeaturizer, "smilesvec", "smilesvec", ("v1", "v2"), id="smilesvec"),
]


@pytest.mark.parametrize(
    ("featurizer_cls", "tunables"),
    [
        (FingerprintsFeaturizer, {"radius", "n_bits", "use_chirality", "use_counts"}),
        (ChemBertaFeaturizer, {"pooling", "max_length"}),
        (SmilesVecDrugFeaturizer, {"k"}),
    ],
)
def test_hyperparameter_space_declares_the_tunables(featurizer_cls: type, tunables: set[str]) -> None:
    assert set(featurizer_cls.get_hyperparameter_space()) == tunables


@pytest.mark.parametrize("featurizer_cls", [p.values[0] for p in _EMBEDDERS])
def test_compute_from_source_without_smiles_raises(featurizer_cls: type) -> None:
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="no SMILES available"):
        featurizer_cls()._compute_from_source(source, np.array(["d1"]))


@pytest.mark.parametrize(("featurizer_cls", "view", "block", "names"), _EMBEDDERS)
def test_transform_blocks_are_named_after_the_featurizer(featurizer_cls, view, block, names) -> None:
    source = MockFeatureSource(features={"d1": {view: np.array([0.1, 0.2])}}, meta_info={view: list(names)})
    ids = np.array(["d1"], dtype=str)

    blocks = featurizer_cls().fit(source, entity_ids=ids).transform_blocks(source, ids)

    assert set(blocks) == {block}
    assert blocks[block].feature_names == names


def test_view_drug_featurizer_is_registered_with_the_morgan_view_default() -> None:
    assert get_drug_featurizer("view") is ViewDrugFeaturizer
    assert ViewDrugFeaturizer.side == "drug"
    assert ViewDrugFeaturizer.input_views == ("morgan_fingerprint",)
    assert ViewDrugFeaturizer()._view == "morgan_fingerprint"
    assert issubclass(ViewDrugFeaturizer, DenseViewFeaturizer)


def test_view_drug_featurizer_passes_the_view_through() -> None:
    source = MockFeatureSource(
        features={
            "d1": {"morgan_fingerprint": np.array([1.0, 0.0])},
            "d2": {"morgan_fingerprint": np.array([0.0, 1.0])},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2"]},
    )
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = ViewDrugFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert featurizer.output_dim == 2
    assert set(blocks) == {"morgan_fingerprint"}
    assert blocks["morgan_fingerprint"].feature_names == ("fp1", "fp2")
    np.testing.assert_allclose(featurizer.transform(source, ids), [[1.0, 0.0], [0.0, 1.0]])


def _generator(n_bits: int = 16):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)


def test_fingerprints_report_a_missing_rdkit(monkeypatch: pytest.MonkeyPatch, drug_source: DrugFeatureSource) -> None:
    block_imports(monkeypatch, "rdkit")

    with pytest.raises(ImportError, match="rdkit is required"):
        FingerprintsFeaturizer(n_bits=16)._compute_from_source(drug_source, drug_source.identifiers[:1])


@pytest.mark.parametrize(
    "smiles", [None, "", float("nan"), "this is not a molecule"], ids=["missing", "empty", "nan", "unparseable"]
)
def test_unusable_smiles_produce_a_nan_fingerprint_row(smiles: object) -> None:
    row = _fingerprint_for_smiles(smiles, _generator(), 16, False)

    assert row.shape == (16,)
    assert np.all(np.isnan(row))


def test_binary_and_count_fingerprints() -> None:
    assert set(np.unique(_fingerprint_for_smiles("CCO", _generator(), 16, False))) <= {0.0, 1.0}
    counts = _fingerprint_for_smiles("CCCCCCCCCC", _generator(8), 8, True)
    binary = _fingerprint_for_smiles("CCCCCCCCCC", _generator(8), 8, False)
    assert counts.sum() > binary.sum()


def test_fingerprints_emit_one_row_per_drug(drug_source: DrugFeatureSource) -> None:
    matrix = FingerprintsFeaturizer(n_bits=32, use_counts=True)._compute_from_source(
        drug_source, drug_source.identifiers[:2]
    )

    assert matrix.shape == (2, 32)


_HIDDEN = torch.tensor([[[1.0, 2.0], [3.0, 8.0], [5.0, 2.0]]])


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("cls", [1.0, 2.0]), ("max", [5.0, 8.0]), ("mean", [3.0, 4.0]), ("not-a-strategy", [3.0, 4.0])],
)
def test_chemberta_pool_applies_the_configured_strategy_and_falls_back_to_mean(
    pooling: str, expected: list[float]
) -> None:
    np.testing.assert_allclose(ChemBertaFeaturizer(pooling=pooling)._pool(_HIDDEN), expected)


class _TinyTokenizer:
    """Stand-in for the Hugging Face tokenizer: one token id ``1..n`` per SMILES character."""

    def __init__(self) -> None:
        self.max_lengths: list[int] = []

    def __call__(self, smiles: str, *, return_tensors: str, truncation: bool, max_length: int) -> dict:
        assert (return_tensors, truncation) == ("pt", True)
        self.max_lengths.append(max_length)
        return {"input_ids": torch.arange(1, min(len(smiles), max_length) + 1).unsqueeze(0)}


class _TinyModel:
    """Stand-in for the encoder: hidden states ``(token id, 1)`` per token, so pooling is predictable."""

    config = SimpleNamespace(hidden_size=2)

    def __init__(self) -> None:
        self.evaluating = False

    def eval(self) -> _TinyModel:
        self.evaluating = True
        return self

    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = torch.stack([input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32)], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def tiny_chemberta(monkeypatch: pytest.MonkeyPatch) -> tuple[_TinyTokenizer, _TinyModel]:
    tokenizer, model = _TinyTokenizer(), _TinyModel()
    monkeypatch.setattr(chemberta, "load_chemberta", lambda: (tokenizer, model))
    return tokenizer, model


def _smiles_lengths(source: DrugFeatureSource, ids: np.ndarray) -> list[int]:
    smiles = get_smiles_for_entities(source, ids)
    assert smiles is not None
    return [len(smiles[drug_id]) for drug_id in ids]


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("mean", lambda n: [(n + 1) / 2, 1.0]), ("cls", lambda n: [1.0, 1.0]), ("max", lambda n: [float(n), 1.0])],
)
def test_chemberta_embeds_each_drug_with_the_configured_pooling(
    tiny_chemberta, drug_source: DrugFeatureSource, pooling, expected
) -> None:
    ids = drug_source.identifiers[:2]

    matrix = ChemBertaFeaturizer(pooling=pooling, max_length=256)._compute_from_source(drug_source, ids)

    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, [expected(n) for n in _smiles_lengths(drug_source, ids)])


def test_chemberta_truncates_to_max_length_and_fills_unknown_drugs_with_nan(
    tiny_chemberta, drug_source: DrugFeatureSource
) -> None:
    tokenizer, _ = tiny_chemberta
    ids = np.array([drug_source.identifiers[0], "not-a-drug"])

    matrix = ChemBertaFeaturizer(pooling="max", max_length=3)._compute_from_source(drug_source, ids)

    assert tokenizer.max_lengths == [3]
    np.testing.assert_allclose(matrix[0], [3.0, 1.0])
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix[1]).all()


@pytest.fixture
def stub_chemberta_weights(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[SimpleNamespace]:
    """Stand in for ``transformers`` and the artifact download; ``load_chemberta`` is cached, so reset it around."""
    calls = SimpleNamespace(tokenizer=[], model=[], tokenizer_obj=_TinyTokenizer(), model_obj=_TinyModel())

    def tokenizer_from_pretrained(path: str, **kwargs: object) -> _TinyTokenizer:
        calls.tokenizer.append((path, kwargs))
        return calls.tokenizer_obj

    def model_from_pretrained(path: str, **kwargs: object) -> _TinyModel:
        calls.model.append((path, kwargs))
        return calls.model_obj

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_from_pretrained)
    transformers.AutoModel = SimpleNamespace(from_pretrained=model_from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(artifacts, "get_artifact_dir", lambda name, filenames: tmp_path)
    calls.directory = str(tmp_path)
    load_chemberta.cache_clear()
    yield calls
    load_chemberta.cache_clear()


def test_load_chemberta_loads_the_local_artifact_once_and_puts_the_model_in_eval_mode(stub_chemberta_weights) -> None:
    calls = stub_chemberta_weights

    first = load_chemberta()
    second = load_chemberta()

    assert first == (calls.tokenizer_obj, calls.model_obj) == second
    assert calls.model_obj.evaluating
    assert calls.tokenizer == [(calls.directory, {"local_files_only": True})]
    assert calls.model == [(calls.directory, {"local_files_only": True})]


def test_load_chemberta_wraps_unexpected_failures_with_guidance(
    monkeypatch: pytest.MonkeyPatch, stub_chemberta_weights
) -> None:
    def broken(name: str, filenames: object) -> None:
        raise ValueError("corrupt weights")

    monkeypatch.setattr(artifacts, "get_artifact_dir", broken)

    with pytest.raises(RuntimeError, match="Could not load ChemBERTa weights.*corrupt weights.*DREVALPY_ARTIFACTS_URI"):
        load_chemberta()


@pytest.mark.parametrize("error", [RuntimeError("download failed"), ImportError("no torch")])
def test_load_chemberta_lets_runtime_and_import_errors_through(
    monkeypatch: pytest.MonkeyPatch, stub_chemberta_weights, error: Exception
) -> None:
    def failing(name: str, filenames: object) -> None:
        raise error

    monkeypatch.setattr(artifacts, "get_artifact_dir", failing)

    with pytest.raises(type(error)) as raised:
        load_chemberta()

    assert raised.value is error


def test_load_chemberta_reports_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    load_chemberta.cache_clear()
    block_imports(monkeypatch, "transformers")

    with pytest.raises(ImportError, match="transformers and torch are required"):
        load_chemberta()

    load_chemberta.cache_clear()


@pytest.mark.network
def test_chemberta_embeds_dataset_smiles(drug_source: DrugFeatureSource) -> None:
    matrix = ChemBertaFeaturizer(max_length=64)._compute_from_source(drug_source, drug_source.identifiers[:2])

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32


class _FakeKeyedVectors:
    def __init__(self, vectors: dict[str, np.ndarray], vector_size: int) -> None:
        self._vectors = vectors
        self.vector_size = vector_size

    @property
    def key_to_index(self) -> dict[str, int]:
        return {word: i for i, word in enumerate(self._vectors)}

    def __getitem__(self, word: str) -> np.ndarray:
        return self._vectors[word]


def test_smilesvec_embed_averages_known_kmers_and_handles_short_and_unknown_smiles() -> None:
    kv = _FakeKeyedVectors({"CCO": np.array([1.0, 3.0]), "CON": np.array([3.0, 5.0]), "CC": np.array([2.0, 4.0])}, 2)

    embedding = _smilesvec_embed("CCOCON", kv, k=3, dim=2)

    np.testing.assert_allclose(embedding, [2.0, 4.0])
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(_smilesvec_embed("CC", kv, k=8, dim=2), [2.0, 4.0])
    np.testing.assert_allclose(_smilesvec_embed("NNNNN", kv, k=3, dim=2), [0.0, 0.0])


@pytest.fixture
def loaded_word2vec(monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[tuple[str, bool]]:
    """Stand in for gensim and the model download; returns the ``(path, binary)`` of every load."""
    loaded: list[tuple[str, bool]] = []
    vectors = {"AB": np.array([1.0, 3.0]), "BC": np.array([3.0, 5.0])}

    def load_word2vec_format(path: str, binary: bool) -> _FakeKeyedVectors:
        loaded.append((path, binary))
        return _FakeKeyedVectors(vectors, 2)

    models = ModuleType("gensim.models")
    models.KeyedVectors = SimpleNamespace(load_word2vec_format=load_word2vec_format)
    monkeypatch.setitem(sys.modules, "gensim", ModuleType("gensim"))
    monkeypatch.setitem(sys.modules, "gensim.models", models)
    monkeypatch.setattr(smilesvec, "get_artifact", lambda name: tmp_path / name)
    return loaded


def test_smilesvec_averages_the_kmer_vectors_of_each_drug(loaded_word2vec, tmp_path) -> None:
    source = SmilesSource({"d1": "ABCD", "d2": "XXXX", "d3": None})
    ids = np.array(["d1", "d2", "d3"])

    matrix = SmilesVecDrugFeaturizer(k=2)._compute_from_source(source, ids)

    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], [2.0, 4.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0])
    assert np.isnan(matrix[2]).all()
    assert loaded_word2vec == [(str(tmp_path / smilesvec._SMILESVEC_MODEL_FILE), False)]


def test_smilesvec_reports_a_missing_gensim(monkeypatch: pytest.MonkeyPatch, drug_source: DrugFeatureSource) -> None:
    block_imports(monkeypatch, "gensim")

    with pytest.raises(ImportError, match="gensim is required"):
        SmilesVecDrugFeaturizer()._compute_from_source(drug_source, drug_source.identifiers[:1])


@pytest.mark.network
def test_smilesvec_embeds_dataset_smiles(drug_source: DrugFeatureSource) -> None:
    matrix = SmilesVecDrugFeaturizer()._compute_from_source(drug_source, drug_source.identifiers[:2])

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32
