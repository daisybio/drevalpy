"""Literature package boundary, facade names, lifecycle ownership and frozen reference metadata."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from drevalpy.components.predictors.literature._metadata import (
    DIPK_REFERENCE,
    DRUGGNN_REFERENCE,
    ENSEMBLEMF_REFERENCE,
    MOLIR_REFERENCE,
    PACCMANN_REFERENCE,
    PHARMAFORMER_REFERENCE,
    PRECILY_REFERENCE,
    SPARSEGO_REFERENCE,
    SRMF_REFERENCE,
    SUPERFELTR_REFERENCE,
)
from drevalpy.models import ModelSpec
from drevalpy.models.config import ModelConfig, from_spec, validate
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.predictor import get as get_predictor

REPO_ROOT = Path(__file__).resolve().parents[4]
PREDICTORS_ROOT = REPO_ROOT / "src" / "drevalpy" / "components" / "predictors"
LITERATURE_ROOT = PREDICTORS_ROOT / "literature"

LITERATURE_FACTORY_NAMES = [
    "DrugGNN",
    "DIPK",
    "MOLIR",
    "SuperFELTR",
    "PharmaFormer",
    "Precily",
    "SRMF",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "SparseGO",
    "EnsembleMF",
    "PaccMann",
]

FORBIDDEN_TOKENS = (
    "literature.impl",
    "LiteratureEngineBase",
    "LiteratureEngineMixin",
    "ENGINE_MODULES",
    "_engine_class_name",
    "raw_engine_adapter",
    "block_engine_adapter",
    "structured_engine_adapter",
)

LITERATURE_PACKAGES = (
    "dipk",
    "sparsego",
    "molir",
    "superfeltr",
    "pharmaformer",
    "precily",
    "srmf",
    "druggnn",
    "ensemblemf",
    "paccmann",
)

LIFECYCLE_METHODS = ("_fit", "_predict", "get_state", "set_state", "is_fitted")


def _python_files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_literature_package_has_no_lazy_predictor_reexports() -> None:
    init_path = LITERATURE_ROOT / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    assert "_LAZY_EXPORTS" not in text
    assert "__getattr__" not in text


def test_literature_tree_has_no_forbidden_engine_indirection() -> None:
    offenders: list[str] = []
    for path in _python_files_under(LITERATURE_ROOT):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, "\n".join(offenders)


def _defined_lifecycle_methods(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined_methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_methods.add(item.name)
    return defined_methods


#: MOLIR and SuperFELTR inherit the lifecycle from the shared per-drug omics shell.
SHARED_LIFECYCLE_OWNER = LITERATURE_ROOT / "_single_drug_omics.py"
SHELL_PACKAGES = frozenset({"molir", "superfeltr"})


@pytest.mark.parametrize("package", LITERATURE_PACKAGES)
def test_literature_predictor_modules_own_lifecycle(package: str) -> None:
    predictor_path = LITERATURE_ROOT / package / "predictor.py"
    assert predictor_path.is_file()
    defined_methods = _defined_lifecycle_methods(predictor_path)
    if package in SHELL_PACKAGES:
        defined_methods |= _defined_lifecycle_methods(SHARED_LIFECYCLE_OWNER)
    missing = [name for name in LIFECYCLE_METHODS if name not in defined_methods]
    assert not missing, f"{predictor_path} missing lifecycle methods: {missing}"


def test_literature_predictor_modules_avoid_removed_adapter_modules() -> None:
    for module_name in (
        "drevalpy.components.predictors.literature.precily.predictor",
        "drevalpy.components.predictors.literature.druggnn.predictor",
        "drevalpy.components.predictors.neural_network.predictor",
        "drevalpy.components.predictors.literature.dipk.predictor",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = Path(source_path).read_text(encoding="utf-8")
        assert "public_models" not in text
        assert "literature.impl" not in text
        assert "LiteratureEngineMixin" not in text


def test_literature_predictor_lazy_package_import() -> None:
    precily_module = "drevalpy.components.predictors.literature.precily.predictor"
    saved = sys.modules.pop(precily_module, None)
    try:
        cls = get_predictor("dipk")
        assert cls.__name__ == "DIPKPredictor"
        assert precily_module not in sys.modules
    finally:
        if saved is not None:
            sys.modules[precily_module] = saved


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_literature_factory_entries_are_native_facades_sharing_the_zoo_name(name: str) -> None:
    spec = ModelSpec.from_name(name)
    config = from_spec(name)
    validate(config)
    assert isinstance(config, ModelConfig)
    assert spec.name == name


ALL_REFERENCES = (
    pytest.param(DRUGGNN_REFERENCE, id="druggnn"),
    pytest.param(PRECILY_REFERENCE, id="precily"),
    pytest.param(SRMF_REFERENCE, id="srmf"),
    pytest.param(MOLIR_REFERENCE, id="molir"),
    pytest.param(SUPERFELTR_REFERENCE, id="superfeltr"),
    pytest.param(PHARMAFORMER_REFERENCE, id="pharmaformer"),
    pytest.param(DIPK_REFERENCE, id="dipk"),
    pytest.param(SPARSEGO_REFERENCE, id="sparsego"),
    pytest.param(ENSEMBLEMF_REFERENCE, id="ensemblemf"),
    pytest.param(PACCMANN_REFERENCE, id="paccmann"),
)

DOIS = (
    pytest.param(DRUGGNN_REFERENCE, "10.1109/TCBB.2021.3060430", id="druggnn"),
    pytest.param(PRECILY_REFERENCE, "10.1038/s41467-022-33291-z", id="precily"),
    pytest.param(SRMF_REFERENCE, "10.1186/s12885-017-3500-5", id="srmf"),
    pytest.param(MOLIR_REFERENCE, "10.1186/s12859-023-05166-7", id="molir"),
    pytest.param(SUPERFELTR_REFERENCE, "10.1186/s12859-023-05166-7", id="superfeltr"),
    pytest.param(PHARMAFORMER_REFERENCE, "10.1038/s41698-025-01082-6", id="pharmaformer"),
    pytest.param(DIPK_REFERENCE, "10.1093/bib/bbae153", id="dipk"),
    pytest.param(SPARSEGO_REFERENCE, "10.1016/j.ebiom.2023.104767", id="sparsego"),
    pytest.param(ENSEMBLEMF_REFERENCE, "10.1038/s41467-026-72903-w", id="ensemblemf"),
    pytest.param(PACCMANN_REFERENCE, "10.1021/acs.molpharmaceut.9b00520", id="paccmann"),
)


@pytest.mark.parametrize("reference", ALL_REFERENCES)
def test_every_reference_has_a_github_repository_and_citation_text(reference: LiteratureReference) -> None:
    assert isinstance(reference, LiteratureReference)
    assert reference.repo_url.startswith("https://github.com/")
    assert reference.citation_text


def test_every_reference_records_a_concrete_model_specific_deviation() -> None:
    deviations = [param.values[0].deviations for param in ALL_REFERENCES]

    assert all(len(note.split()) >= 12 for note in deviations)
    assert len(set(deviations)) == len(deviations)
    assert all("may differ" not in note for note in deviations)


@pytest.mark.parametrize(("reference", "doi"), DOIS)
def test_every_reference_records_its_published_doi(reference: LiteratureReference, doi: str) -> None:
    assert reference.citation_doi == doi


@pytest.mark.parametrize(
    ("reference", "cited"),
    [
        pytest.param(MOLIR_REFERENCE, ("MOLI", "10.1093/bioinformatics/btz318"), id="molir"),
        pytest.param(SUPERFELTR_REFERENCE, ("Super.FELT", "10.1186/s12859-021-04146-z"), id="superfeltr"),
    ],
)
def test_benchmark_doi_references_name_the_original_model_in_the_citation(
    reference: LiteratureReference, cited: tuple[str, str]
) -> None:
    assert all(part in reference.citation_text for part in cited)


def test_repo_urls_are_unique_and_references_immutable() -> None:
    urls = [param.values[0].repo_url for param in ALL_REFERENCES]

    assert len(set(urls)) == len(urls)
    with pytest.raises(AttributeError):
        DIPK_REFERENCE.repo_url = "https://example.invalid"
