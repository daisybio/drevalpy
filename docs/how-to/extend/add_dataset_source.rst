Register a dataset source
=========================

.. interface: both

DrEvalPy loads named datasets (``GDSC1``, ``CTRPv2``, ...) from a **source**: a base URL plus
optional storage settings. You can add your own sources and give names to the dataset files
under them, so that ``drevalpy.load("MyScreen")`` and ``--dataset MyScreen`` work for your
data. This is the way to share a dataset with colleagues through a bucket or a file server. For
a single local file, pass its ``.h5mu`` path directly instead; see
:doc:`/how-to/bring_your_own_data`.

Registration has two levels. You register a source once, then add as many dataset entries under
it as you need. Any protocol that `fsspec <https://filesystem-spec.readthedocs.io/>`_ supports
works as a source URL, as long as its backend is installed. DrEvalPy installs the ``s3`` and
``http`` backends; for ``gs://`` or ``az://`` URLs, install the matching fsspec package
(``gcsfs`` or ``adlfs``) yourself. Local directories work through ``file://`` URLs.

Register a source and a dataset
-------------------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Describe sources and datasets in a YAML file in an extension directory (see
      :doc:`load_extensions`). A YAML file with a ``sources`` or ``datasets`` key is read as a
      dataset declaration; any other YAML file is read as a set of model presets.

      Create ``my_datasets/datasets.yaml``:

      .. code-block:: yaml

         sources:
           my_s3_bucket:
             url: s3://my-bucket/datasets/
             storage_options:
               key: YOUR_ACCESS_KEY
               secret: YOUR_SECRET_KEY
         datasets:
           MyScreen:
             source: my_s3_bucket
             file: MyScreen.h5mu

      Load the directory once with the global ``-e`` option. Any command works; this one also
      shows the result.

      .. snippet: skip

      .. code-block:: bash

         drevalpy -e my_datasets list datasets

   .. tab-item:: Python
      :sync: python

      Call the two registration functions from the public ``drevalpy.registry`` package:

      .. code-block:: python

         from drevalpy import registry

         registry.dataset.register_source(
             "my_s3_bucket",
             "s3://my-bucket/datasets/",
             storage_options={"key": "YOUR_ACCESS_KEY", "secret": "YOUR_SECRET_KEY"},
         )
         registry.dataset.register_dataset("MyScreen", source="my_s3_bucket", file="MyScreen.h5mu")

``storage_options`` is passed to fsspec unchanged, so it takes whatever the protocol takes. Both
calls are **persistent**: the entries are written to ``datasets.json`` in the DrEvalPy
configuration directory (``DREVALPY_CONFIG_DIR``; see :doc:`/reference/environment`) and survive
across sessions, unlike registrations of featurizers and predictors. Because that file is plain
text, do not store secrets in ``storage_options`` on a shared machine. Leave them out and let
fsspec find credentials the way the protocol normally does, for example from the AWS
environment variables or profile for ``s3://``.

Registering the same entry again is a no-op. Registering a different entry under a name that is
already taken, including a built-in name such as ``GDSC1``, raises ``ValueError``; pass
``override=True`` to replace your own entry. A dataset whose ``source`` is unknown raises
``KeyError``.

Check what is registered
------------------------

Entries are saved, so commands after the first registration do not need ``-e`` any more.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      The ``Origin`` column tells built-in datasets from your own (``custom``).

      .. code-block:: bash

         drevalpy list datasets
         drevalpy list datasets MyScreen

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         assert "MyScreen" in registry.dataset.list()

Load a registered dataset
-------------------------

Load by name as usual. DrEvalPy downloads the file once and keeps a copy in its cache directory
(``DREVALPY_CACHE_DIR``). This example needs real credentials and a real bucket, so it is not
run when the documentation is tested.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Pass the name wherever a dataset is expected, for example
      ``drevalpy run MeanBaseline --dataset MyScreen --split-mode LSO``.

   .. tab-item:: Python
      :sync: python

      .. snippet: skip

      .. code-block:: python

         import drevalpy

         dataset = drevalpy.load("MyScreen")

Try it offline with a local directory
-------------------------------------

A ``file://`` source behaves like any other and needs no account, which makes it a good way to
check a declaration before you point it at a bucket. This example saves a synthetic dataset to a
folder, registers the folder as a source, and loads the dataset by name. The cache directory is
redirected only so that the example leaves your normal cache alone.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      Declare the folder with an absolute ``file://`` URL:

      .. code-block:: yaml

         sources:
           lab_files:
             url: file:///data/screens/
         datasets:
           LocalScreen:
             source: lab_files
             file: LocalScreen.h5mu

      Create the file the declaration points at with
      ``drevalpy data synthetic --output /data/screens/LocalScreen.h5mu``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import os
         from pathlib import Path
         from unittest import mock

         import drevalpy
         from drevalpy.testing import build_synthetic_dataset

         screens = Path("screens").resolve()
         screens.mkdir(exist_ok=True)
         build_synthetic_dataset().save(screens / "LocalScreen.h5mu")

         registry.dataset.register_source("lab_files", screens.as_uri())
         registry.dataset.register_dataset("LocalScreen", source="lab_files", file="LocalScreen.h5mu")

         with mock.patch.dict(os.environ, {"DREVALPY_CACHE_DIR": "cache"}):
             dataset = drevalpy.load("LocalScreen")

Next steps
----------

- Use the dataset in an experiment: :doc:`/how-to/run_experiment`.
- Make registrations load automatically in every session: :doc:`load_extensions`.
