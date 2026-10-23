"""Sample featurizer registry: register, discover, and retrieve sample featurizer classes.

The implementation lives in :mod:`drevalpy.registry.featurizer`; this package binds its
sample-side registry to the module-level helpers every other registry package offers.
"""

from drevalpy.registry.featurizer import SampleFeaturizerRegistry, sample_featurizer_registry

__all__ = [
    "SampleFeaturizerRegistry",
    "sample_featurizer_registry",
    "get",
    "list",
    "metadata",
    "register",
    "table",
]

register = sample_featurizer_registry.register
get = sample_featurizer_registry.get
list = sample_featurizer_registry.list_names
table = sample_featurizer_registry.to_dataframe
metadata = sample_featurizer_registry.get_metadata
