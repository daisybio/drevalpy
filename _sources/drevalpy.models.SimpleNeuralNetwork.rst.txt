Simple Neural Network
===========================================

.. _flexible-inputs-simplenn:

Flexible Input System
--------------------------------------------

The baseline neural network models support **flexible inputs**. Rather than hardcoding which omic data type a model uses,
you configure ``cell_line_views`` and ``drug_views`` directly in the ``hyperparameters.yaml`` file.

By doing this, we have replaced the ``ChemBERTaNeuralNetwork`` whose only difference to the ``SimpleNeuralNetwork`` was
its usage of ChemBERTa embeddings instead of fingerprints as input.

Configuring the input views
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default ``SimpleNeuralNetwork`` configuration uses gene expression and fingerprints:

.. code-block:: yaml

    SimpleNeuralNetwork:
        cell_line_views:
            - gene_expression
        drug_views:
            - fingerprints
        dropout_prob:
            - 0.3
        units_per_layer:
            - - 32
              - 16
              - 8
              - 4
        ...

To train the same ``SimpleNeuralNetwork`` with **ChemBERTa** embeddings instead, change ``drug_views``:

.. code-block:: yaml

    SimpleNeuralNetwork:
        cell_line_views:
            - gene_expression
        drug_views:
            - drug_chemberta_embeddings
        dropout_prob:
            - 0.3
        units_per_layer:
            - - 32
              - 16
              - 8
              - 4
        ...

For more, see the documentation of the sklearn models: :ref:`flexible-inputs`.

Simple Neural Network Model
------------------------------------------------------------------

.. automodule:: drevalpy.models.SimpleNeuralNetwork.simple_neural_network
   :members:
   :undoc-members:
   :show-inheritance:

Multi-OMICS Neural Network
----------------------------------------------------------------------

.. automodule:: drevalpy.models.SimpleNeuralNetwork.multi_view_neural_network
   :members:
   :undoc-members:
   :show-inheritance:

Model utils
------------------------------------------------

.. automodule:: drevalpy.models.SimpleNeuralNetwork.utils
   :members:
   :undoc-members:
   :show-inheritance:
