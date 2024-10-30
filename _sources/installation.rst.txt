.. highlight:: shell

Installation
============

DrEvalPy can be installed on all three major platforms (Linux, MacOS, Windows).

Install Python
~~~~~~~~~~~~~~

DrEvalPy requires python >=3.9 and <=3.11. Best practice is to use a clean conda environment (`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_).
Follow the installation guide for your operating system, then create a new environment using

.. code-block:: bash

   conda create -y -n drevalpy python==3.10


Optional dependencies
~~~~~~~~~~~~~~~~~~~~~


Installing DrEvalPy
~~~~~~~~~~~~~~~~~~~~~~

DrEvalPy is listed on the Python Package Index (PyPI) and can be installed with pip. Activate your conda environment (or skip this if you use a system wide python installation) and install the package (and optionally jupyterlab) using

.. code-block:: bash

   conda activate drevalpy
   pip install drevalpy jupyterlab
