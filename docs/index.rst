DrEvalPy
========

DrEvalPy evaluates drug response prediction models. It trains models on sample and
drug data, scores them on held-out pairs, and compares them against baselines under
fixed split modes.

You can drive it from the ``drevalpy`` command line or from Python. Both interfaces run
the same evaluation pipeline, and the how-to guides and tutorials show each task once,
with a tab for each interface. DrEvalPy v2 is a rewrite: if you used 1.x, start with
:doc:`how-to/migrate/index`, because nothing that 1.x wrote loads in v2.

For large or strictly reproducible benchmarks, use the Nextflow pipeline
`nf-core/drugresponseeval <https://nf-co.re/drugresponseeval/dev/>`_. The source code is
on `GitHub <https://github.com/daisybio/drevalpy>`_, the package on
`PyPI <https://pypi.org/project/drevalpy/>`_ and the
`paper <https://doi.org/10.1038/s41467-026-72903-w>`_ on Nature Communications.

Where to go
-----------

- **Learn by doing:** :doc:`tutorials/index` walks you through a first benchmark on a
  synthetic dataset.
- **Get a task done:** :doc:`how-to/index` has recipes for installing DrEvalPy, running,
  tuning and reporting experiments, extending DrEvalPy, and migrating from 1.x.
- **Look something up:** :doc:`reference/index` lists commands, Python APIs, components,
  file formats and the :doc:`glossary <reference/glossary>`.
- **Understand the design:** :doc:`explanation/index` explains why split modes, components
  and registries work the way they do.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   how-to/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   project/contributing
   project/citing
   project/news
   project/contributors
   project/memes
