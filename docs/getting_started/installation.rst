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

Hyperparameter tuning on Windows
--------------------------------

Experiment-time HPO depends on `Ray <https://pypi.org/project/ray/>`_.
Unfortunately, Ray only publishes Windows wheels for Python 3.10-3.12.
This means, if you are using Windows and a Python version outside of that range, the following will happen:

1. Installation of DrEvalPy will succeed, but ``ray`` won't be installed.
2. Running a workflow with hyperparameter tuning (which is enabled by default) will fail with ``ImportError: Ray Tune with Optuna requires ray[tune] and optuna to be installed``.
3. Running a workflow without hyperparameter tuning (``--no_hyperparameter_tuning``) will succeed

To run HPO from Windows make sure to use a supported Python version (3.10-3.12), use
`WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_, or the
:ref:`Docker image <getting_started/installation:With Docker>` above.
