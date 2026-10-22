Install DrEvalPy
================

.. interface: cli

DrEvalPy runs on Linux, macOS and Windows with Python 3.11, 3.12 or 3.13. This page shows how to
install it from source, with pip, in a conda or venv environment, and with Docker. If something
goes wrong, open an issue on `GitHub <https://github.com/daisybio/drevalpy/issues>`_.

.. note::

   DrEvalPy v2 is not released yet. Until it is, ``pip install drevalpy`` installs version 1.6.0,
   which has a different API; see :doc:`/how-to/migrate/index`. Install v2 from source as shown
   below. The pip, conda, venv and Docker instructions apply once v2 is released.

Choose a PyTorch extra
----------------------

PyTorch is an optional extra. Pick the one that matches your machine:

``cpu``
   PyTorch for the CPU. Works everywhere.

``cu126`` or ``cu130``
   PyTorch for a CUDA 12.6 or 13.0 GPU.

The three extras are mutually exclusive. Without any of them, everything that needs PyTorch,
such as the neural-network predictors, is unavailable; the other predictors and all the
benchmarking tools work.

Install from source
-------------------

Clone the repository and let `uv <https://docs.astral.sh/uv/>`_ create the environment. ``uv
sync`` installs the project together with its locked ``dev`` dependency group:

.. snippet: skip

.. code-block:: bash

   git clone https://github.com/daisybio/drevalpy.git
   cd drevalpy
   uv sync --extra cpu

Use ``--extra cu126`` or ``--extra cu130`` instead for a CUDA build. ``uv`` then selects the
matching PyTorch wheel index. Run commands inside the environment with ``uv run drevalpy ...``,
or activate ``.venv`` first.

Install with pip
----------------

Once v2 is released, install it from PyPI with the extra of your choice:

.. snippet: skip

.. code-block:: bash

   pip install "drevalpy[cpu]"

Install in a conda or venv environment
--------------------------------------

Use a clean environment so that DrEvalPy's dependencies do not clash with other projects. With
`mamba <https://mamba.readthedocs.io/>`_ or conda:

.. snippet: skip

.. code-block:: bash

   mamba create -y -n drevalpy python=3.13
   mamba activate drevalpy
   pip install "drevalpy[cpu]"

With the built-in ``venv`` module:

.. snippet: skip

.. code-block:: bash

   python -m venv drevalpy-env
   source drevalpy-env/bin/activate
   pip install "drevalpy[cpu]"

Use Docker
----------

Each published release also builds an image with the CPU build of PyTorch. The image is
published only when a release is published, so there is no tag that follows the ``main``
branch; ``latest`` is the newest release.

.. snippet: skip

.. code-block:: bash

   docker pull ghcr.io/daisybio/drevalpy:latest

The container starts a ``bash`` shell with ``drevalpy`` on the ``PATH``. Mount a directory to keep
your data and results after the container exits:

.. snippet: skip

.. code-block:: bash

   docker run -it -v "$PWD":/work -w /work ghcr.io/daisybio/drevalpy:latest

Check your installation
-----------------------

The command prints the list of subcommands:

.. code-block:: bash

   drevalpy --help

From source with ``uv``, run it as ``uv run drevalpy --help``.

Where DrEvalPy stores files
---------------------------

Built-in datasets download on first use into a cache directory, and pre-trained model artifacts
such as ChemBERTa weights are fetched on first use into a subdirectory of it. Predictions and
reports go to the output directory you pass to the command. Set ``DREVALPY_CACHE_DIR`` to move
the cache, for example to a larger disk. :doc:`/reference/environment` lists this and the other
environment variables, including the ones for offline and air-gapped runs.

Next steps
----------

:doc:`/tutorials/first_benchmark` runs your first experiment on simulated data, without
downloading anything.
