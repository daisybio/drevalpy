RidgeMLPEnsemble
============================

The unweighted average of two models that use only gene expression (the L1000 landmark genes by default):
a per-drug ridge regression whose penalty is chosen per drug by exact leave-one-out error, and a multi-task MLP
with one output per drug, trained on per-drug standardized responses with early stopping.

.. automodule:: drevalpy.models.RidgeMLPEnsemble.ridge_mlp_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.models.RidgeMLPEnsemble.per_drug_ridge
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: drevalpy.models.RidgeMLPEnsemble.multitask_mlp
   :members:
   :undoc-members:
   :show-inheritance:
