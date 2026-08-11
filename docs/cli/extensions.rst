Extensions
==========

If you are reading this, we assume you are already familiar with this
concept:

- :doc:`/concepts/registries`

DrEvalPy's extension system lets you add custom components (featurizers,
predictors, splitters, visualizations, dataset sources) without modifying the
package itself. This page covers how to load extensions from the CLI; for
how to write them, see :doc:`/python/extensions`.

``--extensions-dir``
--------------------

The global ``--extensions-dir`` / ``-e`` option loads an extensions directory
before any subcommand runs:

.. code-block:: bash

   drevalpy --extensions-dir my_components/ run ToyRidge \
       --dataset TOYv1 --split-mode LCO

All ``.py`` files in the directory are imported (triggering any ``@register``
decorators inside them). All ``.yaml`` files are loaded as model-zoo presets
or dataset source declarations.

You can pass the flag multiple times to load several directories:

.. code-block:: bash

   drevalpy -e my_featurizers/ -e my_zoo/ run MyModel --dataset TOYv1 --split-mode LCO

``DREVALPY_EXTENSIONS_DIR``
---------------------------

Set the environment variable to load extensions automatically without a CLI
flag:

.. code-block:: bash

   export DREVALPY_EXTENSIONS_DIR=my_components/
   drevalpy run ToyRidge --dataset TOYv1 --split-mode LCO

Both the environment variable and the CLI flag can be used together — the CLI
flag directories are loaded after the environment variable directory.

Plugin discovery
----------------

For permanent extensions, package them as a Python package and declare the
``drevalpy.plugins`` entry point group. DrEvalPy discovers them automatically
on import without requiring any CLI flag or environment variable.

In your plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."drevalpy.plugins"]
   my_plugin = "my_plugin.components"

Once installed (``pip install my_plugin``), the custom components are available
in every ``drevalpy`` invocation.

Extension directory layout
--------------------------

A typical extension directory:

.. code-block:: text

   my_extensions/
     custom_featurizer.py    # @register_cell_line_featurizer(...)
     custom_predictor.py     # @register_predictor(...)
     custom_splitter.py      # @register_splitter(...)
     custom_zoo.yaml         # zoo presets referencing the new components

Python files are imported in sorted order (``__init__.py`` is skipped).
YAML files are parsed as zoo presets that map names to already-registered
component stacks.

For full examples of each extension type, see :doc:`/python/extensions`.
