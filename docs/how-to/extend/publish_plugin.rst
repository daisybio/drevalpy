Publish components as a plugin package
======================================

.. interface: python

A plugin is an installable Python package whose components appear in every ``drevalpy`` session
once it is installed. There is no flag to pass and no environment variable to set. Use a plugin
for components you want to share or keep permanently. For components that live in a working
directory, use :doc:`load_extensions` instead.

Lay out the package
-------------------

Put the components in your package and give the package a single module that imports all of
them, so importing it triggers every ``register_*`` decorator:

.. code-block:: text

   my_plugin/
     pyproject.toml
     src/my_plugin/
       __init__.py
       components.py        # imports the modules below
       featurizers.py       # @register_sample_featurizer(...)
       predictors.py        # @register_predictor(...)

.. snippet: skip

.. code-block:: python

   # src/my_plugin/components.py
   from my_plugin import featurizers, predictors  # noqa: F401

Write the components with the public surface described in :doc:`write_featurizer`,
:doc:`write_predictor`, :doc:`write_splitter` and :doc:`write_visualization`: import from
``drevalpy.plugin``, not from deeper module paths.

Declare the entry point
-----------------------

Advertise the module in the ``drevalpy.plugins`` entry point group in your ``pyproject.toml``.
The name on the left is how DrEvalPy refers to your plugin; the value is the module to import:

.. code-block:: toml

   [project]
   name = "my-plugin"
   dependencies = ["drevalpy"]

   [project.entry-points."drevalpy.plugins"]
   my_plugin = "my_plugin.components"

Install the package into the same environment as ``drevalpy`` (for example with
``pip install -e .`` during development). DrEvalPy scans for the entry point group when you
import ``drevalpy.registry`` and imports each advertised module.

Find out why a plugin did not load
----------------------------------

A plugin that raises on import would silently remove every component it registers, and the
symptom would be an unknown-predictor error far from the cause. DrEvalPy therefore records the
failure and logs a warning instead of swallowing it. Query the record from Python:

.. code-block:: python

   from drevalpy.registry import get_failed_plugins, get_loaded_plugins

   loaded = get_loaded_plugins()  # {entry point name: declared object reference}
   failed = get_failed_plugins()  # {entry point name: formatted traceback}
   assert isinstance(loaded, dict) and isinstance(failed, dict)

Or run ``drevalpy list plugins`` from the command line. It lists every declared entry point with
its status (``loaded``, ``failed`` or ``not loaded``). Add ``--traceback`` (``-t``) to print the
failing traceback.

Fail loudly in CI
-----------------

The default is non-fatal, so one broken third-party package cannot stop the CLI for everyone
else. Your plugin's own CI wants the opposite: a plugin that does not load is a failure there.
Fail the job in either of these ways:

- Set ``DREVALPY_STRICT_PLUGINS=1`` (also ``true``, ``yes`` or ``on``). Plugin discovery then
  re-raises the original exception when ``drevalpy.registry`` is imported.
- Run ``drevalpy list plugins --strict``, which exits with status 1 if any plugin failed to
  load.

Test the installed plugin
-------------------------

``check_plugin`` from ``drevalpy.testing`` asserts that the entry point is declared, that its
module imports, and that at least one component registered under your package. Put it in the
plugin's test suite, next to the conformance checks described in :doc:`test_components`:

.. snippet: skip

.. code-block:: python

   from drevalpy.testing import check_plugin


   def test_plugin_is_installed_and_registers_components():
       report = check_plugin("my_plugin")
       assert report.components

It raises ``PluginCheckError`` when the entry point is undeclared, fails to import, or registers
nothing. The last case usually means the entry-point module does not import the modules that hold
the decorators.
