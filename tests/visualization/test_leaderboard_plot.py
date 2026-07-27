"""Smoke tests for leaderboard_plot drawing helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from drevalpy.visualization.leaderboard_plot import draw_leaderboard_panels


def test_draw_leaderboard_panels_smoke() -> None:
    df = pd.DataFrame(
        {
            "algorithm": ["ModelA", "ModelB"],
            "PCC": [0.5, 0.4],
            "PCC_std": [0.01, 0.02],
            "RMSE": [1.0, 1.2],
            "RMSE_std": [0.05, 0.05],
            "is_baseline": [False, True],
        }
    )
    colors = {
        "background": "#ffffff",
        "surface": "#f6f8fa",
        "text": "#1f2328",
        "text_secondary": "#57606a",
        "grid": "#d0d7de",
    }
    fig, axes = plt.subplots(1, 2)
    draw_leaderboard_panels(
        fig,
        axes,
        df,
        y_positions=__import__("numpy").arange(1, -1, -1),
        bar_height=0.65,
        font_adder=0,
        colors=colors,
    )
    plt.close(fig)
