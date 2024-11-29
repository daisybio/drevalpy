"""Module containing all drug response prediction models."""

__all__ = [
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "ElasticNetModel",
    "RandomForest",
    "SVMRegressor",
    "SimpleNeuralNetwork",
    "MultiOmicsNeuralNetwork",
    "MultiOmicsRandomForest",
    "SingleDrugRandomForest",
    "SRMF",
    "GradientBoosting",
    "MOLIR",
    "SuperFELTR",
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
    "DIPKModel",
]

from .baselines.multi_omics_random_forest import MultiOmicsRandomForest
from .baselines.naive_pred import NaiveCellLineMeanPredictor, NaiveDrugMeanPredictor, NaivePredictor
from .baselines.singledrug_random_forest import SingleDrugRandomForest
from .baselines.sklearn_models import ElasticNetModel, GradientBoosting, RandomForest, SVMRegressor
from .DIPK.dipk import DIPKModel
from .drp_model import DRPModel
from .MOLIR.molir import MOLIR
from .SimpleNeuralNetwork.multiomics_neural_network import MultiOmicsNeuralNetwork
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from .SRMF.srmf import SRMF
from .SuperFELTR.superfeltr import SuperFELTR

# SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "SingleDrugRandomForest": SingleDrugRandomForest,
    "MOLIR": MOLIR,
    "SuperFELTR": SuperFELTR,
}

# MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
    "GradientBoosting": GradientBoosting,
    "SRMF": SRMF,
    "DIPK": DIPKModel,
}

# MODEL_FACTORY is used in the pipeline!
MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
