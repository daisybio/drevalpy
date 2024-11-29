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

You need Python 3.10+ and Poetry_ to set up your development environment.

.. _Poetry: https://python-poetry.org/


How to test the project
-----------------------

For this, you need to install nox and nox-poetry:

.. code:: console

   $ pip install nox nox-poetry

Run the full test suite:

.. code:: console

   $ nox

List the available Nox sessions:

.. code:: console

   $ nox --list-sessions

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

.. code:: console

   $ nox --session=tests

Unit tests are located in the ``tests`` directory,
and are written using the pytest_ testing framework.

.. _pytest: https://pytest.readthedocs.io/

How to build and view the documentation
---------------------------------------

This project uses Sphinx_ together with several extensions to build the documentation.

To install all required dependencies for the documentation run:

.. code:: console

    $ pip install -r docs/requirements.txt

Please note that drevalpy itself must also be installed. To build the documentation run:

.. code:: console

    $ make html

from inside the docs folder. The generated static HTML files can be found in the `_build/html` folder.
Simply open them with your favorite browser.

.. _sphinx: https://www.sphinx-doc.org/en/master/

How to submit changes
---------------------

Open a `pull request`_ to submit changes to this project against the ``development`` branch.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains a high code coverage.
- If your changes add functionality, update the documentation accordingly.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

.. code:: console

   $ nox --session=pre-commit -- install

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

.. _pull request: https://github.com/daisybio/drevalpy/pulls
