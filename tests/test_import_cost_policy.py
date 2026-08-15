"""Policy test: importing ``drevalpy`` must not drag in the heavy scientific stack.

``drevalpy/registry/__init__.py`` calls ``register_builtin_components()`` at
import time, which imports every registered ``predictor.py``, featurizer and
visualization. Those modules are therefore on the critical path of
``import drevalpy`` - and of every CLI invocation - so a single module-scope
``import torch`` costs every caller a third of a second whether or not they ever
train a model. Deferring the libraries below took ``import drevalpy`` from
**3.59s to 0.21s** end to end, against a 0.02s bare interpreter - the same
figures recorded in ``AGENTS.md``. (An intermediate measurement, taken once
``pytorch_lightning`` was already deferred, still had the remaining libraries
accounting for 2.0s of a 2.2s import, which is why every entry below earns its
place rather than just the two largest.)

The guard lives at the ``tests/`` root rather than beside any one predictor
because it is a cross-package property of the package surface, like
``test_boundary.py`` and ``test_layering_policy.py``.

Each forbidden module below is only needed inside a ``fit``/``predict``/``compute``
call, so the fix when this fails is a function-local import (plus
``if TYPE_CHECKING:`` for annotations), not an addition to this list. Two
shapes need more than a moved import:

* A class that **subclasses** something from a forbidden library (a
  ``torch.utils.data.Dataset``, an ``sklearn.base.BaseEstimator``) cannot defer
  it - the base has to exist when the ``class`` statement runs. Either drop the
  base class where it contributes nothing (``DataLoader`` accepts any object with
  ``__getitem__``/``__len__``) or move the class into its own private module and
  re-export it lazily, as
  ``featurizers/cell_line/_proteomics_transformer.py`` does.
* A module-scope **side effect** that must happen before the library is imported
  anywhere must stay at module scope. ``xgboost_pred.py`` still calls
  ``_set_xgboost_thread_defaults()`` eagerly: deferring it to the import site let
  a test's own ``importorskip("xgboost")`` win the race and segfaulted the suite.

The ``ImportError`` trap: ``_import_modules`` in ``drevalpy/registry/_builtins.py``
swallows import failures during registration and reports them via
``get_skipped_builtin_modules()``. A deferred import moves such a failure into
the training call instead, where nothing catches it - so this file also asserts
registration stayed clean and that the deferred symbols really do resolve.

One interpreter is spawned for the whole module: all facts about a fresh
``import drevalpy`` share the same pristine entry condition, so they are
collected in a single child and asserted separately.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._trusted_subprocess import run_trusted_python

#: Modules that must not be imported as a side effect of ``import drevalpy``.
#: Measured cost of each in a cold interpreter, for context: ``pytorch_lightning``
#: ~1.2s and ``torch_geometric`` ~0.9s (both pull in ``transformers``, via
#: ``torchmetrics.functional.text`` and ``torch_geometric.llm`` respectively),
#: ``torch`` ~0.33s, ``sklearn`` ~0.39s (it reaches ``scipy.stats`` through
#: ``sklearn.utils``), ``xgboost``/``lightgbm`` ~0.4s each (both via ``sklearn``),
#: ``mudata`` ~0.30s (``anndata`` -> ``dask.array`` + ``zarr``), ``pandas`` ~0.14s,
#: ``plotly`` ~0.13s for the surface the plots use (``graph_objects``, ``colors``,
#: ``subplots`` and ``utils``, most of it ``_plotly_utils.basevalidators``),
#: ``scikit_posthocs`` ~0.21s (``seaborn`` -> ``ipywidgets`` -> ``IPython``),
#: ``matplotlib`` ~0.08s, ``wandb`` ~0.11s and ``optuna`` ~0.07s.
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

#: ``module``/``symbol`` pairs the package now imports lazily inside a method.
#: Deferring an import means a typo here would no longer surface at registration
#: time, only once someone trains that model or draws that plot.
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
    ("drevalpy.components.predictors.literature.sparsego.utils", "load_ontology"),
    ("drevalpy.components.predictors.literature.dipk.gene_expression_encoder", "GeneExpressionEncoder"),
    ("drevalpy.components.predictors.literature.dipk.gene_expression_encoder", "encode_gene_expression"),
    (
        "drevalpy.components.predictors.literature.dipk.gene_expression_encoder",
        "train_gene_expession_autoencoder",
    ),
    (
        "drevalpy.components.featurizers.cell_line._proteomics_transformer",
        "ProteomicsMedianCenterAndImputeTransformer",
    ),
)

#: ``module``/``symbol`` pairs a module still re-exports for compatibility after the
#: symbol moved (or its import was deferred) to keep a heavy library off the startup
#: path. These resolve through a module-level ``__getattr__``, which is exactly the
#: kind of indirection a rename would silently break.
LAZY_RE_EXPORTS = (
    (
        "drevalpy.components.featurizers.cell_line.normalized_proteomics",
        "ProteomicsMedianCenterAndImputeTransformer",
    ),
    ("drevalpy.components.featurizers.cell_line.dipk_gene_expression", "GeneExpressionEncoder"),
    ("drevalpy.components.featurizers.cell_line.dipk_gene_expression", "encode_gene_expression"),
    ("drevalpy.components.featurizers.cell_line.dipk_gene_expression", "train_gene_expession_autoencoder"),
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
        "n_cell_line_featurizers": len(drevalpy.registry.cell_line_featurizer.list()),
        "n_drug_featurizers": len(drevalpy.registry.drug_featurizer.list()),
    }}))
    """)


@pytest.fixture(scope="module")
def fresh_import_facts() -> dict[str, object]:
    """Import ``drevalpy`` once in a pristine interpreter and report what happened.

    :returns: Mapping with the forbidden modules that leaked into ``sys.modules``,
        the built-in modules registration had to skip, and the registered counts.
    """
    completed = run_trusted_python(_CHILD_SCRIPT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


class TestAFreshImport:
    """Extended tier: the shared ``fresh_import_facts`` fixture spawns an interpreter.

    All tests read one child-process report, so the ~0.6s is only saved when the
    whole class is deselected. ``test_deferred_training_symbol_resolves`` below needs
    no child process and stays in the fast tier, where it is the cheap half of this
    guard.
    """

    pytestmark = pytest.mark.slow

    def test_import_drevalpy_does_not_import_the_heavy_stack(self, fresh_import_facts: dict[str, object]) -> None:
        assert fresh_import_facts["leaked"] == [], (
            f"import drevalpy pulled in {fresh_import_facts['leaked']}. Move the offending module-scope "
            "import into the method that needs it (and under `if TYPE_CHECKING:` for annotations); "
            "see the docstring of this file."
        )

    def test_import_drevalpy_registers_every_builtin_module(self, fresh_import_facts: dict[str, object]) -> None:
        """A deferred import must not turn into a silently skipped component."""
        assert fresh_import_facts["skipped"] == [], (
            f"registration skipped {fresh_import_facts['skipped']}; their components are unavailable"
        )

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("n_predictors", 27),
            ("n_cell_line_featurizers", 17),
            ("n_drug_featurizers", 10),
        ],
    )
    def test_registration_stayed_eager_and_complete(
        self, fresh_import_facts: dict[str, object], key: str, expected: int
    ) -> None:
        """Making the import cheap must not make registration lazy or partial.

        The point of every deferral in this file is that the *registered module*
        gets cheaper to import, not that fewer modules are registered. Duplicated
        on purpose from ``tests/registry/test_builtins.py``: asserted here it is the
        counter-weight that stops a future "optimisation" from hitting the numbers
        above by simply registering less.
        """
        assert fresh_import_facts[key] == expected


@pytest.mark.parametrize(("module_name", "symbol"), DEFERRED_TRAINING_SYMBOLS)
def test_deferred_training_symbol_resolves(module_name: str, symbol: str) -> None:
    module = __import__(module_name, fromlist=[symbol])
    assert hasattr(module, symbol), f"{module_name} no longer exposes {symbol}"


@pytest.mark.parametrize(("module_name", "symbol"), LAZY_RE_EXPORTS)
def test_lazy_re_export_resolves(module_name: str, symbol: str) -> None:
    """The historical import path must keep working through the lazy re-export."""
    module = __import__(module_name, fromlist=[symbol])
    assert getattr(module, symbol) is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "drevalpy.components.featurizers.cell_line.normalized_proteomics",
        "drevalpy.components.featurizers.cell_line.dipk_gene_expression",
    ],
)
def test_lazy_re_export_still_raises_for_unknown_names(module_name: str) -> None:
    """A module-level ``__getattr__`` must not turn typos into silent ``None``."""
    module = __import__(module_name, fromlist=["__name__"])
    with pytest.raises(AttributeError):
        getattr(module, "definitely_not_a_real_symbol")  # noqa: B009
