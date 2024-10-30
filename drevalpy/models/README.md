This directory is used for the implementation of the models.

Steps required to implement a new model:

1. Create a new file (in a new directory)
2. Create a class for your model which inherits from the DRPModel class (`from dreval.drp_model import DRPModel`)
3. Implement the required methods
4. Add the model to the model factory in `models/__init__.py`
