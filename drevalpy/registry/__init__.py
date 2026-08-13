"""Unified registry module -- all registries, extension loading, plugin discovery."""

from . import cell_line_featurizer as cell_line_featurizer
from . import dataset as dataset
from . import drug_featurizer as drug_featurizer
from . import predictor as predictor
from . import splitter as splitter
from . import visualization as visualization
from ._builtins import get_skipped_builtin_modules as get_skipped_builtin_modules
from ._builtins import register_builtin_components
from ._extensions import (
    load_extension_dir as load_extension_dir,
)
from ._extensions import (
    load_extension_file as load_extension_file,
)
from ._extensions import (
    load_extension_module as load_extension_module,
)
from ._extensions import (
    load_extensions as load_extensions,
)
from ._plugins import discover_plugins
from ._plugins import get_failed_plugins as get_failed_plugins
from ._plugins import get_loaded_plugins as get_loaded_plugins

# Auto-initialize: register builtins, then discover installed plugins
register_builtin_components()
discover_plugins()
