Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `GPL-3.0 license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_

.. _GPL-3.0 license: https://www.gnu.org/licenses/gpl-3.0.html
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
4. Test whether the installation was successful by running a small experiment:

   .. code:: console

      $ uv run drevalpy run NaiveDrugMeanPredictor ElasticNet --dataset GDSC1 --split-mode LCO --no-hpo

5. Visualize the results by running the following command:

   .. code:: console

      $ uv run drevalpy report results/ --output-dir report

   Then open ``report/multiqc_report.html`` in your browser.

How to test the project
-----------------------

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. code:: console

   $ uv run pytest

.. _pytest: https://pytest.readthedocs.io/

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The code must pass all tests.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

Linting and formatting run through `prek <https://github.com/j178/prek>`_, a
drop-in replacement for pre-commit. Run every hook against the whole tree:

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

.. _sphinx: https://www.sphinx-doc.org/en/master/
