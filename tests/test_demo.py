"""Test that the demo notebook runs without errors."""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook_runs():
    """Test that the demo notebook runs without errors."""
    with open("../dreval_colab_demo.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})
