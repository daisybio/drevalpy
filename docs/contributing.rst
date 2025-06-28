Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `GPL-3 license`_ and
highly welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- `Source Code`_
- `Documentation`_
- `Issue Tracker`_

.. _GPL-3 license: https://opensource.org/license/gpl-3-0
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
2. Make a new conda environment with Python 3.11, 3.12, or 3.13.
3. ``pip install poetry`` : we use poetry to manage dependencies
4. ``pip install poetry-plugin-export``
5. ``poetry install`` : this will install all dependencies
6. Test whether the installation was successful by running the following command:

   .. code:: console

      $ drevalpy --run_id my_first_run --models NaiveDrugMeanPredictor ElasticNet --dataset TOYv1 --test_mode LCO

6. Visualize the results by running the following command:

   .. code:: console

      $ drevalpy-report --run_id my_first_run --dataset TOYv1

How to test the project
-----------------------

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The code must pass all tests.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a
Git hook by running the following command:

.. code:: console

   $ nox --session=pre-commit -- install

It is recommended to open an issue before starting work on anything.

.. _pull request: https://github.com/daisybio/drevalpy/pulls

How to build and view the documentation
---------------------------------------

This project uses Sphinx_ together with several extensions to build the documentation.
To build the documentation, change into the docs/ directory and run:

.. code:: console

    $ make html

The generated static HTML files can be found in the `_build/html` folder.
Simply open them with your favorite browser.

.. _sphinx: https://www.sphinx-doc.org/en/master/
