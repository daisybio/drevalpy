"""Module for tensor transformations in the PaccMann model."""


class AugmentTensor:
    """Tensor transformation class."""

    def __init__(self, smiles_language):
        """Initialize the transformation.

        :param smiles_language: SMILES language object
        """
        self.smiles_language = smiles_language

    def __call__(self, tensor):
        """Apply the transformation.

        :param tensor: input tensor
        :return: unchanged tensor
        """
        return tensor
