Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `MIT license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_

.. _MIT license: https://opensource.org/licenses/MIT
.. _Source Code: https://github.com/daisybio/drevalpy
.. _Documentation: https://drevalpy.readthedocs.io/
.. _Issue Tracker: https://github.com/daisybio/drevalpy/issues

How to report a bug
-------------------

Report bugs on the `Issue Tracker`_.


How to request a feature
------------------------

Request features on the `Issue Tracker`_.


How to set up your development environment
------------------------------------------

1. Fork the repository on GitHub.
2. Install `uv <https://docs.astral.sh/uv/>`_, which manages both the Python
   interpreter and the dependencies.
3. ``uv sync`` : this creates the virtual environment and installs the project
   with its ``dev`` dependency group. The ``docs`` group is pulled in on demand
   with ``--group docs``.
4. Test whether the installation was successful by running a small experiment on a simulated
   dataset, which needs no download:

   .. code:: console

      $ uv run drevalpy data synthetic --output toy.h5mu
      $ uv run drevalpy run NaiveDrugMeanPredictor NaiveMeanEffectsPredictor --dataset toy.h5mu --split-mode LSO --no-hpo

5. Visualize the results by running the following command:

   .. code:: console

      $ uv run drevalpy report results/ --output-dir report

   Then open ``report/Drug-Response-Evaluation_multiqc_report.html`` in your browser.

How to test the project
-----------------------

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework. ``tests/`` follows the package's
directory tree, with one test file per feature area rather than per module.

The suite is tiered. The bare command runs the **fast tier** (the vast majority
of tests, finishing in seconds), which is also what the commit hook runs:

.. code:: console

   $ uv run pytest

The remaining tests carry markers and are deselected by default:

``slow``
   The extended tier: tests that spawn a fresh interpreter, fit dose-response
   curves, or train models. A test belongs here if it costs 0.2s or more.

``network``
   Tests that download pretrained weights or remote annotations from the
   artifacts bucket, which is not readable without credentials. These run on a
   weekly schedule in CI.

A ``-m`` on the command line replaces the default marker expression rather than
adding to it, so run the extended tier — or both tiers, which is what CI does on
every pull request — like this:

.. code:: console

   $ uv run pytest -m slow
   $ uv run pytest -m "not network"

To see how many tests each tier holds, count them without running them:

.. code:: console

   $ uv run pytest --collect-only -q
   $ uv run pytest --collect-only -q -m slow
   $ uv run pytest --collect-only -q -m network

Coverage is measured and enforced in CI on the full ``-m "not network"`` run,
as an aggregate floor (``[tool.coverage.report].fail_under`` in ``pyproject.toml``).
The commit hook does not measure it, because the floor is only meaningful on the full
suite. To reproduce the CI check locally:

.. code:: console

   $ uv run pytest -m "not network" --cov --cov-report=term-missing

Run the suite serially. ``pytest-xdist`` is deliberately not a dependency: it was
measured at best ~11% faster on the full coverage run and 9–70% *slower* on the
fast tier, because a cheap ``import drevalpy`` (0.21s, down from 3.59s) left
parallelism nothing to hide. More importantly, ``--dist loadfile``,
``--dist loadscope`` and random ordering are **not** safe yet: the component
registries are process-global, so a test that registers a component without
evicting it again changes what a later test sees. If you introduce parallelism or
shuffling, audit every test that loads an extension or registers a component for
cleanup first.

.. _pytest: https://pytest.readthedocs.io/

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The code must pass all tests.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

Linting and formatting run through `prek <https://github.com/j178/prek>`_, a
drop-in replacement for pre-commit. Run every hook against the whole tree — all
hooks take about 22 seconds, most of it the fast test tier:

.. code:: console

   $ uv run prek run --all-files

To run the same checks automatically on each commit, install the Git hook:

.. code:: console

   $ uv run prek install

It is recommended to open an issue before starting work on anything.

.. _pull request: https://github.com/daisybio/drevalpy/pulls

How to build and view the documentation
---------------------------------------

This project uses Sphinx_ together with several extensions to build the documentation.
Build it from the repository root with warnings treated as errors, exactly as CI does:

.. code:: console

    $ uv run --group docs sphinx-build -W docs docs/_build

The generated static HTML files can be found in ``docs/_build``.
Simply open them with your favorite browser.

The tests in ``tests/docs`` check the layout and run the code on the tutorial and how-to pages:

.. code:: console

    $ uv run pytest tests/docs

.. _sphinx: https://www.sphinx-doc.org/en/master/

How to write documentation
--------------------------

The documentation follows the Diátaxis_ modes. Each page belongs to exactly one section of
``docs/``, and ``tests/docs/test_docs_structure.py`` checks the layout rules below.

``tutorials/``
   A lesson that the reader follows on a synthetic dataset, with the output they should see.
``how-to/``
   One task, named by a title that starts with a verb. ``how-to/extend/`` covers extending
   DrEvalPy and ``how-to/migrate/`` covers moving from 1.x.
``reference/``
   Facts only: no steps and no advice. The command-line, component, model zoo and example
   pages include files that ``docs/conf.py`` generates from the code.
``explanation/``
   Background and design decisions, with no ``python`` or ``bash`` code blocks.
``project/``
   This guide, the citation, the news and the contributors.

Use the terms of ``GLOSSARY.md`` at the repository root. It is the single source of the
glossary page. Write "sample" for the biological source and "pair" for one sample-drug
measurement. The glossary also lists the words to avoid.

A tutorial or how-to page declares which interfaces it covers in a comment on its second
line: ``.. interface: both`` (the default), ``.. interface: python`` or
``.. interface: cli``. On a ``both`` page, every ``python`` or ``bash`` block sits in a
``tab-set`` with ``:sync-group: interface`` and a ``cli`` and a ``python`` tab.

``tests/docs/test_snippets.py`` runs every ``python`` and ``bash`` code block on the tutorial
and how-to pages. The ``python`` blocks of a page share one namespace and run in order in a
temporary directory, and every line of a ``bash`` block must start with ``drevalpy``. A comment
on the line before a block changes how it runs:

``.. snippet: skip``
   Never run the block, for example an install command.
``.. snippet: network``
   Run the block only under ``pytest -m network``, for example one that downloads a
   built-in dataset.

Docstrings
~~~~~~~~~~

Docstrings in ``src/`` use Sphinx field lists, and ``pydoclint`` checks them against the
signature. Describe the parameters with ``:param name:``, the return value with ``:returns:`` and
the exceptions that the function raises itself with ``:raises Type:``:

.. code-block:: python

   def scale(values: np.ndarray, factor: float) -> np.ndarray:
       """Multiply the values by a factor.

       :param values: Response values of one fold.
       :param factor: Multiplier; must be positive.
       :returns: A new array with the scaled values.
       :raises InvalidInputError: If the factor is not positive.
       """

Put types in the annotations only, not in the docstring. Every field must say something that the
name and the type do not. Leave a field out rather than write a placeholder such as
``:returns: Result.`` or ``:param x: x.``; a one-line docstring is enough for a private helper.

.. _Diátaxis: https://diataxis.fr/
