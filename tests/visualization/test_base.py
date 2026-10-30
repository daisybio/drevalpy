"""Tests for the plot interface (:mod:`drevalpy.visualization.base`) and ``PlotRequirement``."""

from __future__ import annotations

import base64
import sys
import types

import pytest
from matplotlib.figure import Figure

from drevalpy.errors import DrEvalError
from drevalpy.types.results import ExperimentResult
from drevalpy.visualization.base import (
    Plot,
    Section,
    embedded_png_html,
    is_matplotlib_figure,
    save_png,
    show,
)
from drevalpy.visualization.requirements import PlotRequirement

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _figure() -> Figure:
    figure = Figure(figsize=(1, 1))
    figure.add_subplot().plot([0, 1], [0, 1])
    return figure


class _StubImagePlot(Plot):
    name = "stub_image"

    def __init__(self) -> None:
        self.drawn: list[object] = []

    def figure(self, result: object) -> Figure:
        self.drawn.append(result)
        return _figure()


class _RecordingFigure:
    """Stands in for a Plotly figure: anything that is not a Matplotlib figure."""

    def __init__(self) -> None:
        self.written: list[str] = []
        self.shown = 0

    def write_image(self, path: str) -> None:
        self.written.append(path)

    def show(self) -> None:
        self.shown += 1


class _StubPlotlyPlot(Plot):
    name = "stub_plotly"

    def __init__(self) -> None:
        self.last = _RecordingFigure()

    def figure(self, result: object) -> _RecordingFigure:
        return self.last


@pytest.fixture
def displayed(monkeypatch: pytest.MonkeyPatch) -> list:
    """Install a recording ``IPython.display.display`` replacing every real ``IPython*`` entry.

    The stub needs ``version_info`` and ``get_ipython`` because matplotlib probes them once per process
    when the first canvas is created.
    """
    recorded: list = []
    ipython = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")
    display_mod.display = recorded.append  # type: ignore[attr-defined]
    ipython.display = display_mod  # type: ignore[attr-defined]
    ipython.version_info = (9, 15, 0, "")  # type: ignore[attr-defined]
    ipython.get_ipython = lambda: None  # type: ignore[attr-defined]
    for name in [n for n in sys.modules if n == "IPython" or n.startswith("IPython.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    return recorded


def _decoded_png(content: str) -> bytes:
    return base64.b64decode(content.split("base64,", 1)[1].split('"', 1)[0])


def test_section_defaults_and_value_equality():
    section = Section(name="Some plot", anchor="some_plot")
    sentinel = object()

    assert (section.description, section.plot, section.content) == ("", None, None)
    assert Section(name="n", anchor="a", description="d", plot=sentinel).plot is sentinel
    assert Section(name="n", anchor="a") == Section(name="n", anchor="a")


class TestPlotContract:
    def test_only_figure_is_abstract_and_the_defaults_describe_an_unconditional_experiment_plot(self):
        with pytest.raises(TypeError):
            Plot()  # type: ignore[abstract]
        assert Plot.__abstractmethods__ == frozenset({"figure"})
        assert (Plot.name, Plot.result_type, Plot.requirements) == ("", ExperimentResult, frozenset())

    def test_default_sections_embed_the_figure_as_one_png_named_after_the_plot(self):
        plot = _StubImagePlot()
        result = object()

        (section,) = plot.sections(result)

        assert plot.drawn == [result]
        assert (section.name, section.anchor, section.plot) == ("stub_image", "stub_image", None)
        assert _decoded_png(section.content).startswith(PNG_MAGIC)

    def test_a_plot_keeps_no_state_between_results(self):
        plot = _StubImagePlot()

        first, second = plot.figure("a"), plot.figure("b")

        assert first is not second
        assert plot.drawn == ["a", "b"]


def test_embedded_png_html_wraps_a_decodable_png_in_an_img_data_uri():
    html = embedded_png_html(_figure())

    assert html.startswith('<img src="data:image/png;base64,')
    assert _decoded_png(html).startswith(PNG_MAGIC)


def test_is_matplotlib_figure_tells_the_two_figure_kinds_apart():
    assert is_matplotlib_figure(_figure())
    assert not is_matplotlib_figure(_RecordingFigure())


class TestMatplotlibRendering:
    @pytest.mark.parametrize("as_str", [False, True])
    def test_save_png_writes_a_png_file_from_a_path_or_a_string(self, tmp_path, as_str):
        out = tmp_path / "out.png"

        save_png(_StubImagePlot(), None, str(out) if as_str else out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_show_draws_the_result_and_hands_the_figure_to_ipython_display(self, displayed):
        plot = _StubImagePlot()

        show(plot, "result")

        assert plot.drawn == ["result"]
        assert len(displayed) == 1
        assert isinstance(displayed[0], Figure)


class TestPlotlyRendering:
    def test_save_png_writes_with_a_string_path_and_show_delegates_to_the_figure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("drevalpy.visualization.base.plotly_png_supported", lambda: True)
        plot = _StubPlotlyPlot()

        save_png(plot, None, tmp_path / "out.png")
        show(plot, None)

        assert plot.last.written == [str(tmp_path / "out.png")]
        assert plot.last.shown == 1

    def test_save_png_raises_where_plotly_cannot_render_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr("drevalpy.visualization.base.plotly_png_supported", lambda: False)
        plot = _StubPlotlyPlot()

        with pytest.raises(DrEvalError, match="not supported on Windows"):
            save_png(plot, None, tmp_path / "out.png")
        assert plot.last.written == []


def test_declares_exactly_the_four_known_capabilities_with_distinct_values():
    assert [r.name for r in PlotRequirement] == ["MULTIPLE_MODELS", "MULTIPLE_FOLDS", "RANDOMIZATION", "ROBUSTNESS"]
    assert len({r.value for r in PlotRequirement}) == len(list(PlotRequirement))


def test_is_hashable_so_plots_can_declare_frozensets():
    requirements = frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})

    assert PlotRequirement.MULTIPLE_MODELS in requirements
    assert PlotRequirement.RANDOMIZATION not in requirements
