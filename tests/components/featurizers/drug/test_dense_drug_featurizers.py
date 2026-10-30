"""Tests for the dense single-view drug featurizers: view, fingerprints, ChemBERTa and SMILESVec.

Only the real weight downloads are ``network``-marked; the loaders are exercised offline with tiny stand-ins.
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from drevalpy.components.featurizers.drug import chemberta, smilesvec
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.drug.chemberta import ChemBertaFeaturizer, load_chemberta
from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
from drevalpy.components.featurizers.drug.smilesvec import SmilesVecDrugFeaturizer
from drevalpy.components.featurizers.drug.view import ViewDrugFeaturizer
from drevalpy.data import artifacts
from drevalpy.registry.featurizer import featurizers
from drevalpy.types.data.feature_source import FeatureSource
from tests._import_shims import block_imports
from tests.components.featurizers.drug._helpers import DrugSource

_EMBEDDERS = [
    pytest.param(FingerprintsFeaturizer, "morgan_fingerprint", "fingerprints", ("fp1", "fp2"), id="fingerprints"),
    pytest.param(ChemBertaFeaturizer, "chemberta", "chemberta", ("e1", "e2"), id="chemberta"),
    pytest.param(SmilesVecDrugFeaturizer, "smilesvec", "smilesvec", ("v1", "v2"), id="smilesvec"),
]


def _fit_transform(featurizer, source: FeatureSource, ids: np.ndarray):
    return featurizer.fit(source, ids).transform(source, ids)


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
def test_a_source_without_the_view_or_smiles_raises(featurizer_cls: type) -> None:
    source = DrugSource({"other": {"d1": [1.0]}})

    with pytest.raises(ValueError, match="no SMILES available"):
        _fit_transform(featurizer_cls(), source, np.array(["d1"]))


@pytest.mark.parametrize(("featurizer_cls", "view", "block", "names"), _EMBEDDERS)
def test_a_stored_view_is_read_and_named_after_the_featurizer(featurizer_cls, view, block, names) -> None:
    source = DrugSource({view: {"d1": [0.1, 0.2]}}, names={view: names})
    ids = np.array(["d1"], dtype=str)

    result = _fit_transform(featurizer_cls(), source, ids)

    assert set(result.blocks) == {block}
    assert result.blocks[block].feature_names == names
    assert featurizer_cls.output_blocks({})[0].name == block
    np.testing.assert_array_equal(result.matrix, np.array([[0.1, 0.2]], dtype=np.float32))


@pytest.mark.parametrize(("featurizer_cls", "view", "block", "names"), _EMBEDDERS)
def test_a_stored_view_is_read_in_fitted_column_order(featurizer_cls, view, block, names) -> None:
    ids = np.array(["d1"], dtype=str)
    fitted = featurizer_cls().fit(DrugSource({view: {"d1": [0.1, 0.2]}}, names={view: names}), ids)

    reordered = DrugSource({view: {"d1": [0.2, 0.1]}}, names={view: names[::-1]})

    np.testing.assert_array_equal(fitted.transform(reordered, ids).matrix, np.array([[0.1, 0.2]], dtype=np.float32))


def test_view_drug_featurizer_is_registered_for_the_drug_side_with_the_morgan_view_default() -> None:
    assert featurizers.get("view") is ViewDrugFeaturizer
    assert ViewDrugFeaturizer.sides == frozenset({"drug"})
    assert ViewDrugFeaturizer.input_views == ("morgan_fingerprint",)
    assert ViewDrugFeaturizer.output_blocks({"view": "chemberta"})[0].name == "chemberta"


def test_view_drug_featurizer_passes_the_view_through() -> None:
    source = DrugSource(
        {"morgan_fingerprint": {"d1": [1.0, 0.0], "d2": [0.0, 1.0]}},
        names={"morgan_fingerprint": ("fp1", "fp2")},
    )
    ids = np.array(["d1", "d2"], dtype=str)

    result = _fit_transform(ViewDrugFeaturizer(), source, ids)

    assert set(result.blocks) == {"morgan_fingerprint"}
    assert result.blocks["morgan_fingerprint"].feature_names == ("fp1", "fp2")
    np.testing.assert_allclose(result.matrix, [[1.0, 0.0], [0.0, 1.0]])


def test_fingerprints_report_a_missing_rdkit(monkeypatch: pytest.MonkeyPatch, smiles_only: DrugSource) -> None:
    block_imports(monkeypatch, "rdkit")

    with pytest.raises(ImportError, match="rdkit is required"):
        _fit_transform(FingerprintsFeaturizer(n_bits=16), smiles_only, smiles_only.identifiers[:1])


def test_unusable_smiles_produce_nan_fingerprint_rows() -> None:
    source = DrugSource(
        smiles={"ok": "CCO", "missing": None, "empty": "", "nan": float("nan"), "bad": "not a molecule"}
    )
    ids = source.identifiers

    matrix = _fit_transform(FingerprintsFeaturizer(n_bits=16), source, ids).matrix

    assert matrix.shape == (5, 16)
    assert not np.isnan(matrix[0]).any()
    assert np.isnan(matrix[1:]).all()


def test_binary_and_count_fingerprints() -> None:
    source = DrugSource(smiles={"d1": "CCCCCCCCCC"})
    ids = source.identifiers

    binary = _fit_transform(FingerprintsFeaturizer(n_bits=8), source, ids).matrix
    counts = _fit_transform(FingerprintsFeaturizer(n_bits=8, use_counts=True), source, ids).matrix

    assert set(np.unique(binary)) <= {0.0, 1.0}
    assert counts.sum() > binary.sum()


def test_fingerprints_are_computed_from_the_smiles_in_one_rdkit_pass_per_fit_and_transform(
    monkeypatch: pytest.MonkeyPatch, smiles_only: DrugSource
) -> None:
    from rdkit.Chem import rdFingerprintGenerator

    passes: list[int] = []
    original = rdFingerprintGenerator.GetMorganGenerator

    def counting_generator(**kwargs: object):
        passes.append(1)
        return original(**kwargs)

    monkeypatch.setattr(rdFingerprintGenerator, "GetMorganGenerator", counting_generator)
    ids = smiles_only.identifiers[:3]

    featurizer = FingerprintsFeaturizer(n_bits=32, use_counts=True).fit(smiles_only, ids)
    assert passes == []
    matrix = featurizer.transform(smiles_only, ids).matrix

    assert passes == [1]
    assert matrix.shape == (3, 32)


def test_fingerprints_read_the_stored_view_without_rdkit(
    monkeypatch: pytest.MonkeyPatch, drug_source: FeatureSource, synthetic_dataset
) -> None:
    block_imports(monkeypatch, "rdkit")
    ids = drug_source.identifiers[:2]

    matrix = _fit_transform(FingerprintsFeaturizer(), drug_source, ids).matrix

    np.testing.assert_array_equal(matrix, synthetic_dataset.get_drug_features("morgan_fingerprint", ids))


def test_fitted_fingerprints_pickle_and_transform_identically(drug_source: FeatureSource) -> None:
    ids = drug_source.identifiers
    fitted = FingerprintsFeaturizer().fit(drug_source, ids)

    restored = pickle.loads(pickle.dumps(fitted))  # noqa: S301 -- pickled on this line

    np.testing.assert_array_equal(
        restored.transform(drug_source, ids).matrix, fitted.transform(drug_source, ids).matrix
    )


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


def _smiles_lengths(source: FeatureSource, ids: np.ndarray) -> list[int]:
    smiles = get_smiles_for_entities(source, ids)
    assert smiles is not None
    return [len(smiles[drug_id]) for drug_id in ids]


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("mean", lambda n: [(n + 1) / 2, 1.0]),
        ("cls", lambda n: [1.0, 1.0]),
        ("max", lambda n: [float(n), 1.0]),
        ("not-a-strategy", lambda n: [(n + 1) / 2, 1.0]),
    ],
)
def test_chemberta_embeds_each_drug_with_the_configured_pooling_and_falls_back_to_mean(
    tiny_chemberta, smiles_only: DrugSource, pooling, expected
) -> None:
    ids = smiles_only.identifiers[:2]

    matrix = _fit_transform(ChemBertaFeaturizer(pooling=pooling, max_length=256), smiles_only, ids).matrix

    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, [expected(n) for n in _smiles_lengths(smiles_only, ids)])


def test_chemberta_truncates_to_max_length_and_fills_unknown_drugs_with_nan(
    tiny_chemberta, smiles_only: DrugSource
) -> None:
    tokenizer, _ = tiny_chemberta
    ids = np.array([smiles_only.identifiers[0], "not-a-drug"])

    matrix = _fit_transform(ChemBertaFeaturizer(pooling="max", max_length=3), smiles_only, ids).matrix

    assert tokenizer.max_lengths == [3]
    np.testing.assert_allclose(matrix[0], [3.0, 1.0])
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix[1]).all()


def test_chemberta_transforms_no_drugs_to_an_empty_block(tiny_chemberta, smiles_only: DrugSource) -> None:
    ids = np.array([], dtype=str)

    assert _fit_transform(ChemBertaFeaturizer(), smiles_only, ids).matrix.shape == (0, 2)


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
def test_chemberta_embeds_dataset_smiles(smiles_only: DrugSource) -> None:
    ids = smiles_only.identifiers[:2]

    matrix = _fit_transform(ChemBertaFeaturizer(max_length=64), smiles_only, ids).matrix

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
    source = DrugSource(smiles={"d1": "ABCD", "d2": "XXXX", "d3": None})
    ids = np.array(["d1", "d2", "d3"])

    matrix = _fit_transform(SmilesVecDrugFeaturizer(k=2), source, ids).matrix

    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], [2.0, 4.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0])
    assert np.isnan(matrix[2]).all()
    assert loaded_word2vec == [(str(tmp_path / smilesvec._SMILESVEC_MODEL_FILE), False)]


def test_smilesvec_embeds_a_smiles_shorter_than_k_as_one_word(loaded_word2vec) -> None:
    source = DrugSource(smiles={"d1": "AB"})

    matrix = _fit_transform(SmilesVecDrugFeaturizer(k=8), source, source.identifiers).matrix

    np.testing.assert_allclose(matrix, [[1.0, 3.0]])


def test_smilesvec_reports_a_missing_gensim(monkeypatch: pytest.MonkeyPatch, smiles_only: DrugSource) -> None:
    block_imports(monkeypatch, "gensim")

    with pytest.raises(ImportError, match="gensim is required"):
        _fit_transform(SmilesVecDrugFeaturizer(), smiles_only, smiles_only.identifiers[:1])


@pytest.mark.network
def test_smilesvec_embeds_dataset_smiles(smiles_only: DrugSource) -> None:
    ids = smiles_only.identifiers[:2]

    matrix = _fit_transform(SmilesVecDrugFeaturizer(), smiles_only, ids).matrix

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32


@pytest.mark.parametrize(("featurizer_cls", "view", "block", "names"), _EMBEDDERS)
def test_the_parameter_free_stored_variant_is_wrapped_as_the_block(featurizer_cls, view, block, names) -> None:
    featurizer = featurizer_cls()
    source = DrugSource({view: {"d1": [0.1, 0.2]}}, names={view: names})
    ids = np.array(["d1"], dtype=str)

    result = featurizer.from_stored(source, ids, np.array([[1.0, 2.0]]))

    assert featurizer.stored_params() == {}
    assert result.blocks[block].feature_names == names
    np.testing.assert_array_equal(result.matrix, np.array([[1.0, 2.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("featurizer", "expected"),
    [
        (FingerprintsFeaturizer(radius=3), {"radius": 3}),
        (FingerprintsFeaturizer(n_bits=512, use_counts=True), {"n_bits": 512, "use_counts": True}),
        (ChemBertaFeaturizer(max_length=128), {"max_length": 128}),
        (SmilesVecDrugFeaturizer(k=4), {"k": 4}),
    ],
)
def test_a_non_default_config_looks_up_only_the_stored_variant_with_its_settings(featurizer, expected) -> None:
    assert featurizer.stored_params() == expected
