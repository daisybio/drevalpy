"""Policy test: importing ``drevalpy`` must not drag in the heavy scientific stack.

The fix when this fails is a function-local import (plus ``if TYPE_CHECKING:`` for
annotations), not an addition to the lists below; see "Import cost" in ``AGENTS.md``.
One interpreter is spawned and all facts about a fresh ``import drevalpy`` are read from it.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import pytest

from tests._trusted_subprocess import run_trusted_python

#: Modules that must not be imported as a side effect of ``import drevalpy``.
FORBIDDEN_STARTUP_IMPORTS = (
    "IPython",
    "anndata",
    "lightgbm",
    "matplotlib",
    "mudata",
    "optuna",
    "pandas",
    "plotly",
    "pytorch_lightning",
    "scikit_posthocs",
    "scipy",
    "seaborn",
    "sklearn",
    "torch",
    "torch_geometric",
    "wandb",
    "xgboost",
)

DEFERRED_TRAINING_SYMBOLS = (
    ("drevalpy.components.predictors.neural_network.network", "FeedForwardNetwork"),
    ("drevalpy.components.predictors.literature.druggnn.algorithm", "DrugGNNModule"),
    ("drevalpy.components.predictors.literature.molir.utils", "MOLIModel"),
    ("drevalpy.components.predictors.literature.superfeltr.utils", "SuperFELTEncoder"),
    ("drevalpy.components.predictors.literature.superfeltr.utils", "SuperFELTRegressor"),
    ("drevalpy.components.predictors.literature.superfeltr.utils", "train_superfeltr_model"),
    ("drevalpy.components.predictors.literature.dipk.model_utils", "Predictor"),
    ("drevalpy.components.predictors.literature.pharmaformer.model_utils", "CombinedModel"),
    ("drevalpy.components.predictors.literature.precily.model_utils", "PrecilyNetwork"),
    ("drevalpy.components.predictors.literature.sparsego.algorithm", "SparseGONetwork"),
    ("drevalpy.components.predictors.literature.ensemblemf.data", "split_pairs"),
    ("drevalpy.components.predictors.literature.ensemblemf.network", "MFNet"),
    ("drevalpy.components.predictors.literature.ensemblemf.training", "train_member"),
    ("drevalpy.components.predictors.literature.sparsego.utils", "load_ontology"),
    ("drevalpy.components.predictors.literature.paccmann.network", "PaccMannNetwork"),
    ("drevalpy.components.predictors.literature.dipk.gene_expression_encoder", "GeneExpressionEncoder"),
    ("drevalpy.components.predictors.literature.dipk.gene_expression_encoder", "encode_gene_expression"),
    (
        "drevalpy.components.predictors.literature.dipk.gene_expression_encoder",
        "train_gene_expession_autoencoder",
    ),
    (
        "drevalpy.components.featurizers.sample._proteomics_transformer",
        "ProteomicsMedianCenterAndImputeTransformer",
    ),
)

_CHILD_SCRIPT = textwrap.dedent(f"""
    import json
    import sys

    import drevalpy  # noqa: F401
    from drevalpy.registry._builtins import get_skipped_builtin_modules

    print(json.dumps({{
        "leaked": sorted(name for name in {FORBIDDEN_STARTUP_IMPORTS!r} if name in sys.modules),
        "skipped": sorted(get_skipped_builtin_modules()),
        "n_predictors": len(drevalpy.registry.predictor.list()),
        "n_sample_featurizers": len(drevalpy.registry.sample_featurizer.list()),
        "n_drug_featurizers": len(drevalpy.registry.drug_featurizer.list()),
    }}))
    """)

_LAZY_GROUP_MEAN_SCRIPT = textwrap.dedent("""
    import json
    import sys

    import drevalpy

    before = "sklearn" in sys.modules
    centerer_class = drevalpy.models.response_transformation.GroupMeanCenterer
    print(json.dumps({
        "sklearn_before": before,
        "class_name": centerer_class.__name__,
        "sklearn_after": "sklearn" in sys.modules,
    }))
    """)


@pytest.fixture(scope="module")
def fresh_import_facts() -> dict[str, Any]:
    completed = run_trusted_python(_CHILD_SCRIPT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.slow
def test_a_fresh_import_leaks_no_heavy_module_and_skips_no_builtin(fresh_import_facts: dict[str, Any]) -> None:
    assert fresh_import_facts["leaked"] == []
    assert fresh_import_facts["skipped"] == []
    assert fresh_import_facts["n_predictors"] > 0
    assert fresh_import_facts["n_sample_featurizers"] > 0
    assert fresh_import_facts["n_drug_featurizers"] > 0


@pytest.mark.parametrize(("module_name", "symbol"), DEFERRED_TRAINING_SYMBOLS)
def test_deferred_training_symbol_resolves(module_name: str, symbol: str) -> None:
    module = __import__(module_name, fromlist=[symbol])
    assert hasattr(module, symbol)


@pytest.mark.slow
def test_group_mean_centerer_loads_sklearn_only_on_first_access() -> None:
    completed = run_trusted_python(_LAZY_GROUP_MEAN_SCRIPT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    facts = json.loads(completed.stdout)
    assert facts == {"sklearn_before": False, "class_name": "GroupMeanCenterer", "sklearn_after": True}
