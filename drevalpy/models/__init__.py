"""Module containing all drug response prediction models."""

__all__ = [
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "NaiveTissueMeanPredictor",
    "NaiveTissueDrugMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "ElasticNetModel",
    "RandomForest",
    "SVMRegressor",
    "SimpleNeuralNetwork",
    "MultiOmicsNeuralNetwork",
    "MultiOmicsRandomForest",
    "SingleDrugRandomForest",
    "SingleDrugElasticNet",
    "SingleDrugProteomicsElasticNet",
    "SRMF",
    "GradientBoosting",
    "MOLIR",
    "SuperFELTR",
    "DIPKModel",
    "ProteomicsRandomForest",
    "ProteomicsElasticNetModel",
    "SingleDrugProteomicsRandomForest",
    "DrugGNN",
    "ChemBERTaNeuralNetwork",
]

from .baselines.multi_omics_random_forest import MultiOmicsRandomForest
from .baselines.naive_pred import (
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaivePredictor,
    NaiveTissueDrugMeanPredictor,
    NaiveTissueMeanPredictor,
)
from .baselines.singledrug_elastic_net import SingleDrugElasticNet, SingleDrugProteomicsElasticNet
from .baselines.singledrug_random_forest import SingleDrugProteomicsRandomForest, SingleDrugRandomForest
from .baselines.sklearn_models import (
    ElasticNetModel,
    GradientBoosting,
    ProteomicsElasticNetModel,
    ProteomicsRandomForest,
    RandomForest,
    SVMRegressor,
)
from .DIPK.dipk import DIPKModel
from .drp_model import DRPModel
from .DrugGNN import DrugGNN
from .MOLIR.molir import MOLIR
from .SimpleNeuralNetwork.multiomics_neural_network import MultiOmicsNeuralNetwork
from .SimpleNeuralNetwork.simple_neural_network import ChemBERTaNeuralNetwork, SimpleNeuralNetwork
from .SRMF.srmf import SRMF
from .SuperFELTR.superfeltr import SuperFELTR

# SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "SingleDrugRandomForest": SingleDrugRandomForest,
    "MOLIR": MOLIR,
    "SuperFELTR": SuperFELTR,
    "SingleDrugElasticNet": SingleDrugElasticNet,
    "SingleDrugProteomicsElasticNet": SingleDrugProteomicsElasticNet,
    "SingleDrugProteomicsRandomForest": SingleDrugProteomicsRandomForest,
}

# MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
    "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
    "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
    "GradientBoosting": GradientBoosting,
    "SRMF": SRMF,
    "DIPK": DIPKModel,
    "ProteomicsRandomForest": ProteomicsRandomForest,
    "ProteomicsElasticNet": ProteomicsElasticNetModel,
    "DrugGNN": DrugGNN,
    "ChemBERTaNeuralNetwork": ChemBERTaNeuralNetwork,
}

# MODEL_FACTORY is used in the pipeline!
MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
