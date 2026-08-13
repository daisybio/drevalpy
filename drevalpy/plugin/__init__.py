"""The supported import surface for third-party drevalpy plugins.

Everything a plugin needs to declare a featurizer, predictor, splitter or
visualization is re-exported here, so a plugin imports from exactly one module::

    from drevalpy.plugin import (
        CellLineFeaturizer,
        FeatureFormat,
        register_cell_line_featurizer,
    )

Nothing is defined in this module. Every name is an alias for a symbol that
lives in the package's internal layout, which means that layout stays free to
move: only the aliases below are a compatibility promise. Importing the deep
paths still works, but they are private in the sense that matters - a refactor
may rename them without a deprecation cycle.

The five ``register_*`` aliases point at the per-registry ``register``
decorators, which are all spelled ``register`` in their own modules. Naming them
apart here is what makes several registrations in one module readable, and
removes the ``from ... import register as register_x`` boilerplate every plugin
would otherwise repeat.
"""

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.featurizers.base import Featurizer, HPOStrategy
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer
from drevalpy.registry.drug_featurizer import register as register_drug_featurizer
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.splitter import (
    Splitter,
    SplitValidationError,
    Validation,
)
from drevalpy.registry.splitter import register as register_splitter
from drevalpy.registry.visualization import register as register_visualization
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeatureBlock,
    graph_feature_block,
    merge_feature_blocks,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import (
    CellLineFeatureSource,
    DrugFeatureSource,
    FeatureSource,
)
from drevalpy.types.data.mudatalike import MuDataLike
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks
from drevalpy.types.enums.literature_reference import LiteratureReference
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode
from drevalpy.types.results import ExperimentResult, ModelResult, RunResult
from drevalpy.visualization.base import ImageVisualization, Section, Visualization
from drevalpy.visualization.requirements import PlotRequirement

__all__ = [
    "BlockPredictor",
    "BlockSpec",
    "CellLineFeatureSource",
    "CellLineFeaturizer",
    "Dataset",
    "DrugFeatureSource",
    "DrugFeaturizer",
    "ExperimentResult",
    "FeatureBlock",
    "FeatureContract",
    "FeatureFormat",
    "FeatureFreePredictor",
    "FeatureSource",
    "Featurizer",
    "HPOStrategy",
    "ImageVisualization",
    "LiteratureReference",
    "MatrixPredictor",
    "ModelInputBatch",
    "ModelResult",
    "ModelScope",
    "MuDataLike",
    "PlotRequirement",
    "PredictionMode",
    "Predictor",
    "ResponseBatch",
    "RunResult",
    "Section",
    "SplitMask",
    "SplitMasks",
    "SplitValidationError",
    "Splitter",
    "TrainingContext",
    "Validation",
    "Visualization",
    "graph_feature_block",
    "merge_feature_blocks",
    "metadata_feature_block",
    "numeric_feature_block",
    "ragged_feature_block",
    "register_cell_line_featurizer",
    "register_drug_featurizer",
    "register_predictor",
    "register_splitter",
    "register_visualization",
]
