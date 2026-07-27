"""Layout tests for critical_difference_layout."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from drevalpy.visualization.critical_difference_layout import critical_difference_diagram


def test_critical_difference_diagram_three_algorithms() -> None:
    ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    sig = pd.DataFrame(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    palette = {"A": "red", "B": "green", "C": "blue"}
    fig, ax = plt.subplots()
    artists = critical_difference_diagram(ranks, sig, color_palette=palette, ax=ax)
    assert artists["markers"]
    plt.close(fig)
