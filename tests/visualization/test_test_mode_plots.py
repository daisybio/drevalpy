"""Tests for test_mode_plots with plotting mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from drevalpy.visualization._legacy.test_mode_plots import draw_test_mode_plots


def test_draw_test_mode_plots_returns_algorithms(tmp_path: Path) -> None:
    ev_res = pd.DataFrame(
        {
            "test_mode": ["LPO", "LPO"],
            "rand_setting": ["predictions", "predictions"],
            "algorithm": ["M1", "M2"],
            "MSE": [1.0, 2.0],
            "CV_split": [0, 0],
        }
    )
    with patch("drevalpy.visualization._legacy.test_mode_plots.CriticalDifferencePlot") as cd_mock:
        cd_mock.return_value.draw_and_save = MagicMock()
        with patch("drevalpy.visualization._legacy.test_mode_plots.Violin") as violin_mock:
            violin_mock.return_value.draw_and_save = MagicMock()
            with patch("drevalpy.visualization._legacy.test_mode_plots.Heatmap") as heat_mock:
                heat_mock.return_value.draw_and_save = MagicMock()
                with patch("drevalpy.visualization._legacy.test_mode_plots.CrossStudyTables") as cs_mock:
                    cs_mock.return_value.draw_and_save = MagicMock()
                    algos = draw_test_mode_plots(
                        "LPO",
                        ev_res,
                        None,
                        None,
                        "run1",
                        tmp_path,
                        tmp_path,
                    )
    assert set(algos) == {"M1", "M2"}
