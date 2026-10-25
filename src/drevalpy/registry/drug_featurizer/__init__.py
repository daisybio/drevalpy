"""Drug featurizer registry: register, discover, and retrieve drug featurizer classes.

The implementation lives in :mod:`drevalpy.registry.featurizer`; this package binds its
drug-side registry to the module-level helpers every other registry package offers.
"""

from drevalpy.registry.featurizer import DrugFeaturizerRegistry, drug_featurizer_registry

__all__ = [
    "DrugFeaturizerRegistry",
    "drug_featurizer_registry",
    "get",
    "list",
    "metadata",
    "register",
    "table",
]

register = drug_featurizer_registry.register
get = drug_featurizer_registry.get
list = drug_featurizer_registry.list_names
table = drug_featurizer_registry.to_dataframe
metadata = drug_featurizer_registry.get_metadata
