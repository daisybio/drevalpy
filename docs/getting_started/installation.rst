.. highlight:: shell

Installation
============

DrEvalPy can be installed on all three major platforms (Linux, MacOS, Windows).
If something goes wrong, feel free to open an issue on `GitHub <https://github.com/daisybio/drevalpy/issues>`_.

With pip
--------

DrEvalPy requires Python >=3.11 and is available on PyPI:

.. code-block:: bash

   pip install drevalpy

With Conda
----------

DrEvalPy requires python >=3.11. Best practice is to use a clean
(`mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_) or
conda environment (`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_). Mamba is automatically installed
when downloading (`Miniforge <https://github.com/conda-forge/miniforge>`_) and is generally faster and better at
resolving dependencies.
Follow the installation guide for your operating system, then create a new environment using

.. code-block:: bash

   mamba create -y -n drevalpy python=3.13

Activate your conda environment and install the package using

.. code-block:: bash

   mamba activate drevalpy
   pip install drevalpy

With venv
---------

DrEvalPy can also be installed using the built-in `venv` module. First, create a new environment and activate it:

.. code-block:: bash

   python -m venv drevalpy-env
   source drevalpy-env/bin/activate

Then, install the package using pip:

.. code-block:: bash

   pip install drevalpy

With Docker
-----------

DrEvalPy is available as a `Docker image <ghcr.io/daisybio/drevalpy:main>`_.

Pull the image:

.. code-block:: bash

   docker pull ghcr.io/daisybio/drevalpy:latest

Run the image:

.. code-block:: bash

   docker run -it ghcr.io/daisybio/drevalpy:latest

From Source
-----------

To install DrEvalPy from source, clone the repository and let
`uv <https://docs.astral.sh/uv/>`_ create the environment. ``uv sync`` installs
the project together with the locked ``dev`` dependency group:

.. code-block:: bash

    git clone https://github.com/daisybio/drevalpy.git
    cd drevalpy
    uv sync

Verify the console script:

.. code-block:: bash

   uv run drevalpy --help

Then :doc:`choose the CLI or Python API </getting_started/run_first_experiment>`
for your first experiment. Built-in datasets download on first use into a
system cache directory (``platformdirs.user_cache_dir("drevalpy")``, for
example ``~/.cache/drevalpy`` on Linux or ``~/Library/Caches/drevalpy`` on
macOS). Set the ``DREVALPY_CACHE_DIR`` environment variable to use a
different location. Predictions and reports go under the output directory you
pass to the command (``--output-dir`` / ``-o``, default ``results`` for
``drevalpy run`` and ``report`` for ``drevalpy report``).

Pre-trained model artifacts (ChemBERTa weights, MolGNet checkpoint, PPI
embeddings, ...) are fetched on first use from an object-storage location and
cached under ``<cache dir>/artifacts``. Two environment variables control this:

``DREVALPY_ARTIFACTS_URI``
   Base URI to fetch artifacts from. Any fsspec-supported protocol works, so
   this can point at a mirror bucket or a local directory (useful for offline
   or air-gapped runs).

``DREVALPY_ARTIFACTS_STORAGE_OPTIONS``
   JSON object passed to fsspec, for example ``{"profile": "my-aws-profile"}``
   or ``{"anon": true}``. Unset by default, so the ambient credential chain
   (environment variables, shared AWS config, instance roles) applies.

Hyperparameter tuning on Windows
--------------------------------

Experiment-time HPO depends on `Ray <https://pypi.org/project/ray/>`_.
Unfortunately, Ray only publishes Windows wheels for Python 3.10-3.12.
This means, if you are using Windows and a Python version outside of that range, the following will happen:

1. Installation of DrEvalPy will succeed, but ``ray`` won't be installed.
2. Running a workflow with hyperparameter tuning (which is enabled by default) will fail with ``ImportError: Ray Tune with Optuna requires ray[tune] and optuna to be installed``.
3. Running a workflow without hyperparameter tuning (``--no-hpo``) will succeed

To run HPO from Windows make sure to use a supported Python version (3.10-3.12), use
`WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_, or the
:ref:`Docker image <getting_started/installation:With Docker>` above.
