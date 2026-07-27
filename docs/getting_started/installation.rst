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

To install DrEvalPy from source, clone the repository and install the package using Poetry
(ensure that Poetry is  >=1.2.0 because otherwise, the group dependencies will not work, e.g., 2.4.1 works):

.. code-block:: bash

    git clone https://github.com/daisybio/drevalpy.git
    cd drevalpy
    mamba create -y -n drevalpy python==3.13 poetry==2.4.1
    poetry --version
    pip install poetry-plugin-export
    poetry install

Now, you can test the functionality quickly via ``drevalpy --help``. Or take a look at the
:doc:`/cli/quickstart` documentation.

Ray Tune on Windows
-------------------

Experiment-time hyperparameter search uses **Ray Tune** to schedule and run
trials (including optional CPU/GPU resource limits) and **Optuna**
(``OptunaSearch``) only as the sampler that proposes the next configurations.
There is no Optuna-only path: without Ray, ``hyperparameter_tuning=True``
fails at import time. You can still run experiments with defaults via
``--no_hyperparameter_tuning`` / ``hyperparameter_tuning=False``.

Ray publishes ``win_amd64`` wheels for Python 3.10–3.12, so tuning works on
native Windows with those interpreters. There is still no Windows wheel for
Python 3.13+, so ``ray[tune]`` is skipped at install time in that case (see
``pyproject.toml``). If you need Ray Tune on Windows with Python 3.13+, use
Python 3.12 instead, `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_,
or the :ref:`Docker image <getting_started/installation:With Docker>` above.

Backward compatibility
----------------------

Before 1.6.0, Ray Tune was sometimes documented as an optional ``multiprocessing``
extra, and model libraries such as ``xgboost``, ``lightgbm``, ``gseapy``,
``mygene``, and ``obonet`` were install extras (``drevalpy[xgboost]``,
``drevalpy[precily]``, and so on). Those packages are core dependencies now
(Ray is still skipped on Windows with Python 3.13+, as described above). The
``multiprocessing`` Poetry extra and the model-library extras are no longer
used for that purpose.
