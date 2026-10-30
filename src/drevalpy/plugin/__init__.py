"""The supported import surface for third-party drevalpy plugins.

Everything a plugin needs to write a featurizer, predictor, splitter or plot is
re-exported here, so a plugin imports from exactly one module::

    from drevalpy.plugin import Featurizer, FeaturizerResult, register_featurizer

Nothing is defined in this module. Every name is an alias for a symbol that lives in
the package's internal layout, which stays free to move: only the aliases below are a
compatibility promise. The deep paths still import, but a refactor may rename them.

The four ``register_*`` aliases point at the ``register`` decorators, which are all
spelled ``register`` in their own modules; naming them apart here keeps several
registrations in one module readable. :func:`view_matrix` and :func:`dense_result`
live in the underscore-private ``featurizers/_dense.py``; the alias here is the promise.
The conformance checks for a plugin's own test suite are in :mod:`drevalpy.testing`.
"""

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import FeatureMismatchError, dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.data.quality import CurveQuality, curve_quality_mask
from drevalpy.data.splitters import observed_mask
from drevalpy.errors import DrEvalError, InvalidInputError, UnknownNameError
from drevalpy.registry import load_extensions
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.featurizer import register as register_featurizer
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.splitter import Splitter, SplitValidationError, Validation
from drevalpy.registry.splitter import register as register_splitter
from drevalpy.registry.visualization import register as register_visualization
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeatureBlock,
    FeaturizerResult,
    global_feature_block,
    graph_feature_block,
    merge_feature_blocks,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, Side
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode
from drevalpy.types.results import ExperimentResult, ModelResult
from drevalpy.visualization.base import Plot, Section, save_png, show
from drevalpy.visualization.requirements import PlotRequirement

__all__ = [
    "BlockSpec",
    "CurveQuality",
    "Dataset",
    "DrEvalError",
    "ExperimentResult",
    "FeatureBlock",
    "FeatureFormat",
    "FeatureFreePredictor",
    "FeatureMismatchError",
    "FeatureSource",
    "Featurizer",
    "FeaturizerResult",
    "Inputs",
    "InvalidInputError",
    "LiteratureReference",
    "MatrixPredictor",
    "ModelInputBatch",
    "ModelResult",
    "ModelScope",
    "Plot",
    "PlotRequirement",
    "PredictionMode",
    "Predictor",
    "Section",
    "Side",
    "SplitMask",
    "SplitMasks",
    "SplitValidationError",
    "Splitter",
    "UnknownNameError",
    "Validation",
    "all_nan_rows_mask",
    "curve_quality_mask",
    "dense_result",
    "global_feature_block",
    "graph_feature_block",
    "load_extensions",
    "merge_feature_blocks",
    "numeric_feature_block",
    "observed_mask",
    "ragged_feature_block",
    "register_featurizer",
    "register_predictor",
    "register_splitter",
    "register_visualization",
    "save_png",
    "show",
    "view_matrix",
]
