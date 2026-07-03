.. highlight:: shell

Installation
============

DrEvalPy can be installed on all three major platforms (Linux, MacOS, Windows).
If something goes wrong, feel free to open an issue on `GitHub <https://github.com/daisybio/drevalpy/issues>`_.

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

DrEvalPy is listed on the Python Package Index (PyPI) and can be installed with pip.
Activate your conda environment (or skip this if you use a system wide python installation)
and install the package using

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


Optional Features (Extras)
--------------------------

Some models and features require additional dependencies that are **not** installed by the
default ``pip install drevalpy``. They are provided as optional `extras`:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Extra
     - Enables
     - Extra dependencies
   * - ``precily``
     - The ``Precily`` model (GSVA pathway features)
     - ``gseapy``
   * - ``xgboost``
     - The ``MultiViewXGBoost`` baseline model
     - ``xgboost``
   * - ``multiprocessing``
     - Parallelized cross-validation / tuning via Ray
     - ``ray`` (and ``pydantic``, usually already present)

Install one or more extras by listing them in square brackets, for example:

.. code-block:: bash

   pip install "drevalpy[precily]"
   pip install "drevalpy[precily,xgboost,multiprocessing]"

If you install from source with Poetry, install the extras with ``-E`` (or use
``--all-extras`` to install all of them):

.. code-block:: bash

   poetry install -E precily -E xgboost -E multiprocessing
   # or, equivalently
   poetry install --all-extras


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

Now, you can test the functionality quickly via `drevalpy --help`. Or take a look at the `Quickstart <./quickstart.html>`_ documentation.
