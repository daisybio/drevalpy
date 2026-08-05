"""Policy checks for the restructured Sphinx documentation tree."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
ZOO_DIR = REPO_ROOT / "drevalpy" / "models" / "zoo"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

WORKFLOW_DIRS = (
    DOCS / "cli",
    DOCS / "python",
    DOCS / "concepts",
    DOCS / "getting_started",
)

EXEMPT_SUFFIXES = {
    "index.rst",
    "reference.rst",
    "_generated_reference.rst",
    "_generated_cell_line_featurizers.rst",
    "_generated_drug_featurizers.rst",
    "_generated_model_zoo.rst",
    "_generated_predictors.rst",
}

EXEMPT_RELATIVE = {
    "python/api",
}

CLI_COMMANDS = (
    "viability-preprocess",
    "viability-postprocess",
    "load-response",
    "make-cv-pkls",
    "make-hpam-yamls",
    "train-cv",
    "evaluate-hpams",
    "test-cv",
    "make-randomization-yamls",
    "make-final-split-pkls",
    "tune-final-model",
    "train-final-model",
    "consolidate-single-drug",
    "evaluate-test",
    "collect-results",
    "report",
    "make-pipeline-report",
)


def _rst_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.rst") if path.is_file())


def _is_exempt(path: Path) -> bool:
    rel = path.relative_to(DOCS).as_posix()
    if path.name in EXEMPT_SUFFIXES:
        return True
    return any(rel.startswith(prefix) for prefix in EXEMPT_RELATIVE)


def _workflow_pages() -> list[Path]:
    pages: list[Path] = []
    for directory in WORKFLOW_DIRS:
        for path in _rst_files(directory):
            if not _is_exempt(path):
                pages.append(path)
    return pages


def test_backward_compatibility_sections_are_final_and_substantive() -> None:
    """If a page has Backward compatibility, it must be last and non-empty."""
    misplaced: list[str] = []
    empty: list[str] = []
    for path in _workflow_pages():
        text = path.read_text(encoding="utf-8")
        if "Backward compatibility" not in text:
            continue
        matches = list(
            re.finditer(
                r"^(?P<title>\S.*)\n(?P<underline>-{3,})\s*$",
                text,
                flags=re.MULTILINE,
            )
        )
        rel = path.relative_to(DOCS).as_posix()
        if not matches or matches[-1].group("title").strip() != "Backward compatibility":
            misplaced.append(rel)
            continue
        section = text[matches[-1].start() :]
        if re.search(r"no branch-specific", section, flags=re.IGNORECASE):
            empty.append(f"{rel} (no-op branch note)")
        if re.search(r"under-listed|incorrectly stated", section, flags=re.IGNORECASE):
            empty.append(f"{rel} (docs-only note)")
        if "Before 1.6.0" not in section and "before 1.6.0" not in section:
            empty.append(f"{rel} (missing before 1.6.0)")
    assert not misplaced, "Backward compatibility must be the final top-level section:\n" + "\n".join(misplaced)
    assert not empty, "Empty or docs-only Backward compatibility sections:\n" + "\n".join(empty)


def test_zoo_presets_documented_in_model_zoo() -> None:
    from _model_zoo import generate_model_zoo_rst

    zoo_names = {path.stem for path in ZOO_DIR.glob("*.yaml")}
    catalog = (DOCS / "concepts" / "model_zoo.rst").read_text(encoding="utf-8")
    assert "_generated_model_zoo.rst" in catalog
    generated = generate_model_zoo_rst()
    missing = sorted(name for name in zoo_names if name not in generated)
    assert not missing, f"Zoo presets missing from generated model zoo catalog: {missing}"
    assert "``scaledGeneExpression:singleDrugElasticNet``" in generated
    assert "``scaledGeneExpression:singleDrugRandomForest``" in generated
    assert "``molirOmics:molir``" in generated
    assert "``superfeltrOmics:superfeltr``" in generated


def test_component_catalog_is_registry_driven_and_synchronized() -> None:
    from drevalpy.components import register_builtins as rb
    from drevalpy.components.registry import (
        list_cell_line_featurizer_metadata,
        list_drug_featurizer_metadata,
        list_predictor_metadata,
    )

    catalog = (DOCS / "concepts" / "component_catalog.rst").read_text(encoding="utf-8")
    for include in (
        "_generated_cell_line_featurizers.rst",
        "_generated_drug_featurizers.rst",
        "_generated_predictors.rst",
    ):
        assert f".. include:: {include}" in catalog
    assert ".. list-table::" not in catalog
    assert "Example recipe" not in catalog

    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(DOCS)!r})\n"
        "from _component_catalog import generate_component_catalog_rsts\n"
        "sys.stdout.write(json.dumps(generate_component_catalog_rsts(), sort_keys=True))\n"
    )
    completed = run_trusted_python(script, cwd=str(REPO_ROOT))
    assert completed.returncode == 0, completed.stderr
    generated = json.loads(completed.stdout)

    rb.register_builtin_components()
    expected_featurizers = {
        "cell_line": {
            row["name"]: (row["output_format"], " ".join(row["description"].split()))
            for row in list_cell_line_featurizer_metadata()
            if row["name"] in rb._CELL_LINE_MODULES
        },
        "drug": {
            row["name"]: (row["output_format"], " ".join(row["description"].split()))
            for row in list_drug_featurizer_metadata()
            if row["name"] in rb._DRUG_MODULES
        },
    }
    expected_predictors = {
        row["name"]: (
            row["input_interface"].replace("_", "-").capitalize(),
            " ".join(row["description"].split()),
        )
        for row in list_predictor_metadata()
        if row["name"] in rb._PREDICTOR_MODULES
    }
    for registry_name, expected in expected_featurizers.items():
        matches = re.findall(
            r"^   \* - ``([^`]+)``\n     - ``([^`]+)``\n     - ([^\n]+)$",
            generated[registry_name],
            flags=re.MULTILINE,
        )
        observed = {name: (output_format, description) for name, output_format, description in matches}
        assert observed == expected

    predictor_matches = re.findall(
        r"^   \* - ``([^`]+)``\n     - ([^\n]+)\n     - ([^\n]+)$",
        generated["predictor"],
        flags=re.MULTILINE,
    )
    observed_predictors = {name: (interface, description) for name, interface, description in predictor_matches}
    assert len(expected_featurizers["cell_line"]) == len(rb._CELL_LINE_MODULES) == 17
    assert len(expected_featurizers["drug"]) == len(rb._DRUG_MODULES) == 9
    assert len(expected_predictors) == len(rb._PREDICTOR_MODULES) == 27
    assert observed_predictors == expected_predictors


def test_cli_pages_have_no_python_code_blocks() -> None:
    offenders: list[str] = []
    for path in _rst_files(DOCS / "cli"):
        text = path.read_text(encoding="utf-8")
        if "code-block:: python" in text:
            offenders.append(path.relative_to(DOCS).as_posix())
    assert not offenders, f"CLI pages must not contain Python code blocks: {offenders}"


def test_python_guide_pages_have_no_shell_cli_blocks() -> None:
    offenders: list[str] = []
    for path in _rst_files(DOCS / "python"):
        if path.relative_to(DOCS).as_posix().startswith("python/api"):
            continue
        text = path.read_text(encoding="utf-8")
        if "code-block:: bash" in text or "code-block:: shell" in text:
            offenders.append(path.relative_to(DOCS).as_posix())
        cli_invocation = re.compile(
            r"^\s*drevalpy\s+(?:--|" + "|".join(re.escape(cmd) for cmd in CLI_COMMANDS) + r")\b",
            flags=re.MULTILINE,
        )
        if cli_invocation.search(text):
            offenders.append(path.relative_to(DOCS).as_posix())
    assert not offenders, f"Python guides must not contain CLI shell examples: {offenders}"


def test_concept_pages_have_no_interface_code_blocks() -> None:
    """Concept pages stay interface-neutral except composition notation tabs.

    ``from_components_to_models`` may show ``ModelConfig`` Python snippets as
    one of three equivalent notations (recipe / YAML / ModelConfig), not as a
    Python-API tutorial.
    """
    offenders: list[str] = []
    allowed_python = {"concepts/from_components_to_models.rst"}
    for path in _rst_files(DOCS / "concepts"):
        rel = path.relative_to(DOCS).as_posix()
        text = path.read_text(encoding="utf-8")
        banned = ["code-block:: bash", "code-block:: shell"]
        if rel not in allowed_python:
            banned.append("code-block:: python")
        if any(token in text for token in banned):
            offenders.append(rel)
    assert not offenders, f"Concept pages must stay interface-neutral: {offenders}"


def test_compatibility_wording_avoids_stale_version_labels() -> None:
    stale: list[str] = []
    for path in _workflow_pages():
        text = path.read_text(encoding="utf-8")
        if "1.5.1" in text or "modularity release" in text.lower():
            stale.append(path.relative_to(DOCS).as_posix())
    assert not stale, "Compatibility wording issues:\n" + "\n".join(stale)


def test_cli_reference_documents_all_subcommands() -> None:
    from _cli_click import generate_cli_reference_rst
    from typer.main import get_command

    from drevalpy.cli.main import app

    reference = (DOCS / "cli" / "reference.rst").read_text(encoding="utf-8")
    assert "_generated_reference.rst" in reference

    generated = generate_cli_reference_rst()
    missing_in_docs = [cmd for cmd in CLI_COMMANDS if f"drevalpy {cmd}" not in generated]
    assert not missing_in_docs, f"Generated CLI reference missing commands: {missing_in_docs}"

    click_app = get_command(app)
    names = set(getattr(click_app, "commands", {}))
    missing = sorted(set(CLI_COMMANDS) - names)
    assert not missing, f"Typer app missing expected commands: {missing}"


def test_redirects_cover_legacy_urls() -> None:
    conf_text = (DOCS / "conf.py").read_text(encoding="utf-8")
    required = [
        "usage.html",
        "runyourmodel.html",
        "hyperparameter_migration.html",
        "API.html",
        "reference.html",
        "quickstart.html",
        "installation.html",
    ]
    missing = [url for url in required if url not in conf_text]
    assert not missing, f"Missing redirects for: {missing}"


def test_zoo_yaml_files_are_valid() -> None:
    for path in sorted(ZOO_DIR.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded is not None, path.name


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        (DOCS / "python" / "api" / "index.rst", "Backward compatibility"),
        (DOCS / "cli" / "reference.rst", "Backward compatibility"),
    ],
)
def test_generated_reference_pages_omit_compatibility(path: Path, forbidden: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert forbidden not in text
