Extend DrEvalPy
===============

DrEvalPy is built from registries: names mapped to featurizers, predictors, splitters,
visualizations and dataset sources. You can add your own entries without changing the package.
Write the component with the public ``drevalpy.plugin`` surface, check it with
``drevalpy.testing``, then make it visible either by loading it from a directory or by shipping
it as an installable plugin.

**Write a component**

- :doc:`write_featurizer`: turn raw views or identifiers into features.
- :doc:`write_predictor`: add a model that learns from sample-drug pairs.
- :doc:`write_splitter`: add a split mode with its own leakage rule.
- :doc:`write_visualization`: add a plot that the report picks up.

**Check and share it**

- :doc:`test_components`: run the shipped conformance checks on your components.
- :doc:`load_extensions`: load components and model presets from files and directories.
- :doc:`publish_plugin`: ship components as an installable package.
- :doc:`add_dataset_source`: register your own dataset location.

.. toctree::
   :maxdepth: 1

   write_featurizer
   write_predictor
   write_splitter
   write_visualization
   test_components
   load_extensions
   publish_plugin
   add_dataset_source
