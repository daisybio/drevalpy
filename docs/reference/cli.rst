Command-line reference
======================

The ``drevalpy`` command has the subcommands listed below. The section is generated from the
command definitions.

- Extensions are loaded before a subcommand runs, from ``DREVALPY_EXTENSIONS_DIR`` and then
  from each ``--extensions-dir`` / ``-e``; see :doc:`environment` and :doc:`extension_files`.
- ``drevalpy run`` and ``drevalpy single`` accept zoo names and recipes; see :doc:`model_zoo` and
  :doc:`recipes`. ``drevalpy run`` also builds a grid from ``-S``, ``-D`` and ``-P``.
- The files the commands write are described in :doc:`results`.

.. include:: _generated_cli.rst
