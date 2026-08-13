Extensions
==========

If you are reading this, we assume you are already familiar with:

- :doc:`models` — ``construct_model``, recipes, ``ModelConfig``, and lifecycle
- :doc:`/concepts/component_catalog`
- :doc:`/concepts/from_components_to_models`
- :doc:`/concepts/registries`

Every extensible concept in DrEvalPy — predictors, featurizers, splitters,
datasets, and visualizations — is managed by a registry that maps
human-readable names to implementations. This page shows how to register
custom implementations for each extension point and make them available to the
pipeline.

One import surface: ``drevalpy.plugin``
---------------------------------------

Import everything you need from :mod:`drevalpy.plugin`. It re-exports every
base class, value type, and registration decorator a component needs:

.. code-block:: python

   from drevalpy.plugin import CellLineFeaturizer, FeatureFormat, register_cell_line_featurizer

Nothing is defined there — each name is an alias for a symbol that lives deeper
in the package. That indirection is the point: **only the aliases are a
compatibility promise.** The deep paths still import, but they are private in
the sense that matters, and a refactor may rename them without a deprecation
cycle. The five ``register_*`` aliases point at the per-registry ``register``
decorators, which are all spelled ``register`` in their own modules; naming them
apart is what makes several registrations in one module readable.

DrEvalPy also ships a ``py.typed`` marker, so the annotations on those aliases
are visible to a type checker running over your plugin.

The examples on this page are executed
--------------------------------------

Every snippet below is ``literalinclude``\ d from a real module under
``docs/examples/``. The documentation build imports each of them, runs
DrEvalPy's shipped conformance checks over the result, and compares what landed
in the registries against a pinned list. An example that stops working fails the
build rather than misleading you, and these are the components it registers:

.. include:: _generated_examples.rst

Nothing in the shipped package imports ``docs/examples/``, and the build rolls
the registrations back once it has checked them, so the toy names above are not
present in a normal session.

Custom featurizers
------------------

Subclass ``CellLineFeaturizer`` or ``DrugFeaturizer`` and register with
``@register_cell_line_featurizer`` or ``@register_drug_featurizer``.

What a featurizer must provide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The public ``fit`` / ``transform`` / ``transform_blocks`` methods are
**implemented for you** on the base class: they detect entities whose feature
rows are entirely NaN, run your code on the rest, and splice NaN rows back into
the result. You override the hooks underneath them:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Responsibility
   * - ``_fit(source, *, entity_ids=None, ...)``
     - Learn whatever ``_transform_blocks`` needs, on pre-validated entity ids.
       Return ``self``.
   * - ``_transform_blocks(source, entity_ids)``
     - Return a ``dict[str, FeatureBlock]``, one row per entity id.
   * - ``output_dim`` (property)
     - Feature width after fitting.
   * - ``get_state`` / ``set_state``
     - Optional, but required for a fitted featurizer to survive a ``*.zip``
       checkpoint: those store the state mapping, not the object.

``_transform(source, entity_ids)`` returning a flat matrix is optional; the base
class derives it by concatenating your numeric blocks.

Two declarations are mandatory, and registration is **rejected** without them:

**The feature contract.** A ``FeatureFormat`` (``numeric_matrix``, ``graph``, or
``ragged_sequence``) describing the payload format this featurizer produces.
Composition validation compares it to the predictor's ``cell_line_contract`` /
``drug_contract`` and rejects stacks whose formats disagree. Declare it either
as ``contract`` on the class body or as ``contract=`` on the decorator; when
both are present the decorator argument wins.

**The input views.** Which raw feature views the featurizer reads, so the
data-loading layer never needs a name-to-view lookup table. Any one of these
satisfies it:

- ``input_views = ("gene_expression",)`` — a fixed set of views.
- ``entity_id_only = True`` — no view at all, just the identifiers.
- ``requires_view = True`` — the view comes from the config, as a ``view=``
  construction kwarg.
- overriding ``resolve_input_views`` — the views depend on other
  hyperparameters.

Registration also rejects a class that still has unimplemented abstract
methods, naming the members it is missing. That check runs at registration
rather than at instantiation, so a missing ``_fit`` fails next to its cause.

A view-reading cell-line featurizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: /examples/toy_cell_line_featurizer.py
   :language: python
   :caption: docs/examples/toy_cell_line_featurizer.py

An identifier-only drug featurizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This one declares its contract on the class body instead, and uses
``entity_id_only`` in place of ``input_views``:

.. literalinclude:: /examples/toy_drug_featurizer.py
   :language: python
   :caption: docs/examples/toy_drug_featurizer.py

Custom predictors
-----------------

Every predictor must inherit **exactly one** input interface and register with
``@register_predictor``. The three interfaces were introduced in
:doc:`/concepts/component_catalog`.

As with featurizers, the public ``fit`` / ``predict`` are implemented on the
base class — they reject a batch with no responses, drop pairs whose features
are NaN, and return NaN for those pairs on predict. Contracts may be declared on
the class body or passed to the decorator, and the decorator wins.

.. tab-set::

   .. tab-item:: Feature-free

      ``FeatureFreePredictor`` sees pair identifiers and response values only.
      Composition forbids cell-line and drug featurizers for it, since it would
      consume neither, but registration still wants both contracts because the
      composition checker compares them before it knows the interface.
      Implement ``_fit`` and ``_predict``.

      .. literalinclude:: /examples/toy_mean_predictor.py
         :language: python
         :caption: docs/examples/toy_mean_predictor.py

   .. tab-item:: Matrix

      ``MatrixPredictor`` implements ``_fit`` / ``_predict`` for you by calling
      ``batch.to_feature_matrix()``, so you implement ``_fit_matrix`` /
      ``_predict_matrix`` on the dense pair-level design matrix — the pattern
      ElasticNet, RandomForest and friends use. Both contracts must be
      ``numeric_matrix``; registration rejects anything else for this interface.

      .. literalinclude:: /examples/toy_ridge_predictor.py
         :language: python
         :caption: docs/examples/toy_ridge_predictor.py

   .. tab-item:: Block

      ``BlockPredictor`` reads named featurizer blocks from
      ``batch.cell_line_blocks`` / ``batch.drug_blocks`` instead of one
      flattened matrix. Contracts still constrain the **format** of each side;
      ``required_cell_line_blocks`` / ``required_drug_blocks`` additionally
      require named blocks to be present in the stack. Implement ``_fit`` and
      ``_predict``.

      .. literalinclude:: /examples/toy_block_predictor.py
         :language: python
         :caption: docs/examples/toy_block_predictor.py

Custom splitters
----------------

A splitter is a function, not a class. Register it under a mode name with
``@register_splitter``; it must accept the splitter protocol signature and
return a list of :class:`~drevalpy.types.SplitMasks`.

.. literalinclude:: /examples/toy_splitter.py
   :language: python
   :caption: docs/examples/toy_splitter.py

The ``validation`` argument names the leakage constraint the registry enforces
automatically after every split (``"LCO"``, ``"LDO"``, ``"LPO"``, or ``"LTO"``).
The registry wraps your function, so the check cannot be bypassed by calling the
registered splitter directly; a violation raises ``SplitValidationError``.

Registering a mode name that is already taken **raises**. Pass
``override=True`` when replacing an existing mode is the intent — a silent
overwrite would let one package quietly change another's split semantics.

Once registered, the mode works anywhere a mode string is accepted:

.. code-block:: python

   from drevalpy.data import split

   folds = split(dataset, mode="TOY_LCO", n_splits=5)

Custom visualizations
---------------------

Register a visualization class with ``@register_visualization``.
``Visualization`` declares **four** abstract methods:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Method
     - Responsibility
   * - ``compute(result, dataset=None)``
     - Derive the plot data from the result and store it on the instance.
   * - ``to_png(path)``
     - Write a static image.
   * - ``to_multiqc()``
     - Return ``Section`` objects for the report.
   * - ``show()``
     - Display interactively in a notebook.

For a static Matplotlib plot, subclass ``ImageVisualization`` instead: it
implements the last three in terms of a figure, leaving you ``compute`` — which
must assign the figure to ``self._fig`` — and ``_create_figure``.

.. literalinclude:: /examples/toy_visualization.py
   :language: python
   :caption: docs/examples/toy_visualization.py

``result_type`` declares whether the visualization operates on an
``ExperimentResult`` (aggregated across models) or a ``ModelResult`` (a single
model). ``requirements`` is a frozenset of ``PlotRequirement`` values naming
conditions the report system checks before selecting the plot automatically —
multiple CV folds, multiple models, randomization, or robustness data.

As with splitters, a name that is already registered raises unless you pass
``override=True``.

Testing your components
-----------------------

:mod:`drevalpy.testing` ships in the wheel precisely so a plugin's own test
suite can import it. It removes the two things that otherwise block an offline
plugin CI: every registered dataset lives in a credentialed bucket, and a
predictor needs a fully featurized batch.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Helper
     - What it gives you
   * - ``build_synthetic_dataset(...)``
     - An in-memory ``Dataset``. Response-only by default; pass ``omics=[...]``
       to add the cell-line modalities your featurizer reads.
   * - ``build_synthetic_batch(dataset, ...)``
     - A ``ModelInputBatch`` with drawn features and a learnable response, so a
       predictor is testable without composing a model.
   * - ``observed_pairs(dataset)``
     - Every measured cell-line/drug pair, as a ``ResponseBatch``.
   * - ``FEATURIZER_CHECKS`` / ``PREDICTOR_CHECKS``
     - Every conformance check, as a tuple to parametrize over.
   * - ``check_plugin(name)``
     - Assert an installed plugin's entry point is declared, loaded, and that
       its components resolve through the registries.

Each check takes ``(cls, fixture, **kwargs)`` — the fixture being a dataset for
featurizers and a batch for predictors, and optional in both cases — so a suite
parametrizes over a whole family at once:

.. code-block:: python

   import pytest

   from drevalpy.testing import FEATURIZER_CHECKS, build_synthetic_dataset

   from my_plugin.featurizers import MyFeaturizer


   @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
   def test_my_featurizer_conforms(check):
       check(MyFeaturizer, build_synthetic_dataset(omics=["gene_expression"]))

The checks catch what registration cannot: that the component instantiates,
that ``output_dim`` agrees with the width ``transform`` actually produced, and
that a fresh instance restored from ``get_state`` reproduces the original's
output. The last one is the expensive bug — a fitted attribute left out of
``get_state`` makes a reloaded checkpoint silently predict something else. A
failing check raises ``ConformanceError``, which subclasses ``AssertionError``.

This page's own examples are verified exactly this way; see
``docs/examples/toy_conformance.py`` in the repository.

Custom dataset sources
----------------------

Register remote or local storage locations as **sources**, then point named
datasets at files under those sources:

.. code-block:: python

   from drevalpy.registry.dataset import register_dataset, register_source

   register_source(
       "my_s3_bucket",
       "s3://my-bucket/datasets/",
       storage_options={"key": "...", "secret": "..."},
   )

   register_dataset("MyScreen", source="my_s3_bucket", file="MyScreen.h5mu")

The two-level design means you register a source once and then add as many
dataset entries under it as needed. Any protocol that
`fsspec <https://filesystem-spec.readthedocs.io/>`_ supports works: HTTPS,
S3, GCS, Azure Blob Storage, or local file paths. Unlike the other registries,
dataset entries are persisted to a local configuration file, so they survive
across sessions. Once registered, load by name as usual:

.. code-block:: python

   from drevalpy.data import load

   dataset = load("MyScreen")

Literature references
---------------------

``LiteratureReference`` is optional **provenance metadata** for components
ported from a paper or external repository. Pass it as ``reference=...`` on the
register decorator, as ``toyRidge`` above does. It does **not** change training,
composition checks, or checkpoints — it only documents where the idea came from.
``repo_url`` is required; ``citation_text``, ``citation_doi`` and ``deviations``
are optional strings.

Loading extensions
------------------

Import your components
~~~~~~~~~~~~~~~~~~~~~~

``@register_*`` runs when the module is imported. If your package is
installable (or otherwise on ``PYTHONPATH``), a normal import is enough:

.. code-block:: python

   import my_components.toy_featurizer  # registers toyCellLine
   import my_components.toy_predictors  # registers toyMean, toyRidge, …

   from drevalpy.models import construct_model

   ToyRidge = construct_model(
       "ToyRidge",
       "toyCellLine:toyDrugHash:toyRidge",
   )
   model = ToyRidge()

Other sources: ``load_extensions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~drevalpy.registry.load_extensions` when components are not a
normal importable package, or when you also want to register external zoo
YAML in one call:

- ``modules`` — dotted names (same effect as ``import``)
- ``files`` — individual ``.py`` paths executed as temporary modules
- ``directories`` — all ``*.py`` in a folder (non-recursive; ``__init__.py``
  skipped; sorted by filename)
- ``zoo_files`` — YAML presets that map a **name** to an already-registered
  stack (not Python classes, not experiment hpam YAML)

.. code-block:: python

   from drevalpy.models import construct_model
   from drevalpy.registry import load_extensions

   load_extensions(
       directories=["my_components"],
       zoo_files=["my_zoo/toy.yaml"],
   )
   ToyMean = construct_model("toyMean")  # zoo preset name

Every entry point rolls the registries back if loading fails part-way, so a
module that registers two components and then raises leaves neither behind.

Plugin discovery
~~~~~~~~~~~~~~~~

When the package is imported, it scans for installed Python packages that
advertise the ``drevalpy.plugins`` entry point group. Importing the advertised
module triggers registration decorators, making a plugin's components
available without any explicit user action beyond installation.

In your plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."drevalpy.plugins"]
   my_plugin = "my_plugin.components"

When a plugin fails to load
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plugin that raises on import silently removes every component it would have
registered, which surfaces much later as an unknown-predictor error far from the
cause. DrEvalPy therefore records the failure instead of swallowing it:

.. code-block:: python

   from drevalpy.registry import get_failed_plugins, get_loaded_plugins

   get_loaded_plugins()  # {entry point name: declared object reference}
   get_failed_plugins()  # {entry point name: formatted traceback}

The default is non-fatal, so one broken third-party package cannot brick the
CLI for everyone else. Setting the environment variable
``DREVALPY_STRICT_PLUGINS=1`` re-raises instead — which is what a plugin's own
CI wants, since a plugin that does not load is a failure there rather than a
degraded experience.

Extension directories
~~~~~~~~~~~~~~~~~~~~~

Both the CLI and the Python API accept an **extensions directory** containing
``.py`` and ``.yaml`` files. All Python files in the directory are imported
(triggering registration decorators for any registry), and all YAML files are
loaded as model-zoo presets or dataset declarations. The environment variable
``DREVALPY_EXTENSIONS_DIR`` provides the same mechanism without requiring a
CLI flag.

Saving and loading with custom components
-----------------------------------------

Checkpoints are ZIP archives that store the resolved ``ModelConfig`` (component
**names**) and fitted state — not the Python classes themselves. On load,
DrEvalPy looks those names up in the registries again, then restores state. If
a custom featurizer or predictor is not registered in the process that calls
``load`` / ``load_model``, reconstruction fails.

Import the same modules (or call ``load_extensions``) before loading:

.. code-block:: python

   import my_components.toy_featurizer
   import my_components.toy_predictors

   from drevalpy.models import load_model

   model = load_model("checkpoints/toy_ridge.zip")

Built-in zoo models need no extra step; only custom component names require
this. See :doc:`models` for the general save/load lifecycle.
