Create a plugin
===============

.. interface: python

This page is the workflow for creating a DrEvalPy plugin package, step by step, whether you
follow it yourself or hand it to a coding agent. Each step says how to check that it worked.
The other pages of this section explain the parts in depth; this one puts them in order.

Start from the demo plugin
--------------------------

The `demo plugin <https://github.com/drevalpy/demo-plugin>`__ is a complete plugin package with
one example of each component kind (a sample and a drug featurizer, a predictor, a splitter and
a plot), a ``zoo.yaml`` preset, tests and a CI workflow. Copy it rather than starting from an
empty directory:

.. code-block:: console

   $ git clone https://github.com/drevalpy/demo-plugin my-plugin
   $ cd my-plugin
   $ rm -rf .git && git init
   $ mv src/drevalpy_demo_plugin src/my_plugin

Then rename the plugin everywhere it is named:

- ``name`` in ``pyproject.toml``, for example ``my-plugin``;
- the entry point in ``[project.entry-points."drevalpy.plugins"]``, for example
  ``my_plugin = "my_plugin.components"``;
- the imports of ``drevalpy_demo_plugin`` in ``src/`` and ``tests/``.

Delete the component kinds you do not need, together with their imports in ``components.py``
and their tests. To check, install the package and run its tests; ``drevalpy list plugins``
shows the entry point as ``loaded``:

.. code-block:: console

   $ grep -rn drevalpy_demo_plugin src tests pyproject.toml   # prints nothing
   $ uv sync
   $ uv run pytest
   $ uv run drevalpy list plugins

Read the API of the installed version
-------------------------------------

Write against the drevalpy version installed in the plugin's environment, not against a
separate checkout or another version's documentation, because the two can differ. This prints
the file that defines the plugin surface; read it and the docstrings of the classes it
re-exports:

.. code-block:: console

   $ uv run drevalpy --version
   $ uv run python -c "import drevalpy.plugin, inspect; print(inspect.getsourcefile(drevalpy.plugin))"

Import components only from ``drevalpy.plugin``, and in tests also from ``drevalpy.testing``.
Deeper module paths still import, but they may change in any release.

Choose names
------------

Every name a plugin registers must be unique among the built-ins and every installed plugin:
registering a name twice raises ``ValueError``, so the plugin fails to load. Follow the
style of the built-ins: camelCase for featurizers, predictors and plots (such as
``singleDrugElasticNet``), PascalCase for presets (such as ``SingleDrugElasticNet``) and upper case
for split modes (such as ``LSO``). A prefix of the plugin's own, as the demo's ``demoRidge`` and
``DemoModel``, keeps the names apart. Check what is taken:

.. code-block:: console

   $ uv run drevalpy list predictors
   $ uv run drevalpy list sample-featurizers
   $ uv run drevalpy list drug-featurizers
   $ uv run drevalpy list splitters
   $ uv run drevalpy list visualizations
   $ uv run drevalpy list zoo-presets

Write the components
--------------------

Replace the demo components with yours, one module per kind, as the demo does:

- :doc:`write_featurizer` for sample and drug featurizers;
- :doc:`write_predictor` for predictors. A model that fits one model per drug subclasses
  ``SingleDrugPredictor``, which routes the pairs and fits the drugs in parallel; see
  :ref:`how-to/extend/write_predictor:Fit one model per drug`;
- :doc:`write_splitter` for split modes;
- :doc:`write_visualization` for plots.

Ship presets
------------

The command line selects models by preset name, so give every model a user should run a preset
in a ``zoo.yaml`` next to the components:

.. code-block:: yaml

   MyModel:
     sample_featurizer: myTopGenes
     drug_featurizer: fingerprints
     predictor: myRidge

The entry-point module imports the component modules first, which runs their ``register_*``
decorators, and then loads the presets, which name those components. This is the demo's
``components.py``:

.. snippet: skip

.. code-block:: python

   from pathlib import Path

   from drevalpy.plugin import load_extensions

   from my_plugin import featurizers, plots, predictors, splitters

   # directories=[] stops load_extensions from also reading DREVALPY_EXTENSIONS_DIR.
   load_extensions(zoo_files=[str(Path(__file__).with_name("zoo.yaml"))], directories=[])

   __all__ = ["featurizers", "plots", "predictors", "splitters"]

The preset keys are listed in :doc:`/reference/extension_files` and the featurizer forms in
:doc:`/reference/recipes`. A preset that does not load raises an error that names the entry and
the bad key and suggests close names, for example ``Unknown sample featurizer
'myTopGenez'. Did you mean 'myTopGenes'?``. ``drevalpy list zoo-presets`` shows the preset
once it loads.

Test the plugin
---------------

Keep the demo's three kinds of test: the shipped conformance checks for every featurizer and
predictor, ``check_plugin`` with the entry-point name, and one end-to-end run of a preset on the
synthetic dataset. ``n_splits=2`` keeps the run short, and ``on_error="raise"`` makes a failing
run fail the test instead of being recorded:

.. snippet: skip

.. code-block:: python

   import pytest

   import drevalpy
   from drevalpy import ModelSpec
   from drevalpy.testing import FEATURIZER_CHECKS, PREDICTOR_CHECKS, build_synthetic_batch, check_plugin

   from my_plugin.featurizers import TopGenes
   from my_plugin.predictors import Ridge


   def test_plugin_registers_its_components():
       assert check_plugin("my_plugin").components


   @pytest.mark.parametrize("check", FEATURIZER_CHECKS)
   def test_featurizer_conforms(check):
       check(TopGenes, drevalpy.synthetic())


   @pytest.mark.parametrize("check", PREDICTOR_CHECKS)
   def test_predictor_conforms(check):
       check(Ridge, build_synthetic_batch(drevalpy.synthetic(), sample_block_names=("gene_expression",)))


   def test_preset_runs_end_to_end():
       result = drevalpy.run(
           [ModelSpec.from_name("MyModel")],
           drevalpy.synthetic(),
           "LPO",
           n_splits=2,
           hyperparameter_tuning=False,
           on_error="raise",
       )
       assert result.model_names == ["MyModel"]

:doc:`test_components` describes each check and the synthetic fixtures.

Run CI and publish
------------------

A plugin that fails to import only logs a warning by default. In the plugin's CI, make it an
error: set ``DREVALPY_STRICT_PLUGINS=1`` for the whole job and run
``drevalpy list plugins --strict`` after the tests, as the demo's
``.github/workflows/ci.yml`` does. If a component downloads files, such as pretrained weights
from the Hugging Face Hub, cache them between CI runs instead of downloading them on every job,
for example:

.. code-block:: yaml

   - uses: actions/cache@v4
     with:
       path: ~/.cache/huggingface
       key: huggingface-weights-v1

:doc:`publish_plugin` covers the package layout, the entry point and how to find out why a
plugin did not load.
