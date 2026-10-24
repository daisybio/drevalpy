"""Tests for :mod:`drevalpy.visualization.base`; the shared guards and embedding are covered once via stubs."""

from __future__ import annotations

import base64
import sys
import types

import matplotlib.pyplot as plt
import pytest

from drevalpy.errors import DrEvalError
from drevalpy.visualization.base import (
    ImageVisualization,
    PlotlyVisualization,
    Section,
    Visualization,
    embedded_png_html,
    require_figure,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _StubImagePlot(ImageVisualization):
    registry_name = "stub_image"

    def compute(self, result=None) -> None:
        self._fig = self._create_figure()

    def _create_figure(self):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.plot([0, 1], [0, 1])
        return fig


class _RecordingFigure:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.shown = 0

    def write_image(self, path: str) -> None:
        self.written.append(path)

    def show(self) -> None:
        self.shown += 1


class _StubPlotlyPlot(PlotlyVisualization):
    registry_name = "stub_plotly"

    def compute(self, result=None) -> None:
        self._fig = _RecordingFigure()

    def to_multiqc(self) -> list[Section]:
        return []


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def stub() -> _StubImagePlot:
    return _StubImagePlot()


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


def test_section_defaults_and_value_equality():
    section = Section(name="Some plot", anchor="some_plot")
    sentinel = object()

    assert (section.description, section.plot, section.content) == ("", None, None)
    assert Section(name="n", anchor="a", description="d", plot=sentinel).plot is sentinel
    assert Section(name="n", anchor="a") == Section(name="n", anchor="a")


class TestVisualizationContract:
    def test_the_abstract_surface_of_each_base(self):
        with pytest.raises(TypeError):
            Visualization()  # type: ignore[abstract]
        assert Visualization.__abstractmethods__ == frozenset({"compute", "to_png", "to_multiqc", "show"})
        assert Visualization.registry_name == ""
        assert ImageVisualization.__abstractmethods__ == frozenset({"compute", "_create_figure"})
        assert PlotlyVisualization.__abstractmethods__ == frozenset({"compute", "to_multiqc"})


def test_require_figure_returns_the_figure_or_names_the_caller():
    sentinel = object()

    assert require_figure(sentinel, "to_png") is sentinel
    with pytest.raises(RuntimeError, match=r"Call compute\(\) before some_method\(\)"):
        require_figure(None, "some_method")


def test_embedded_png_html_wraps_a_decodable_png_in_an_img_data_uri(stub):
    stub.compute()

    html = embedded_png_html(stub._fig)

    assert html.startswith('<img src="data:image/png;base64,')
    assert base64.b64decode(html.split("base64,", 1)[1].split('"', 1)[0]).startswith(PNG_MAGIC)


class TestImageVisualization:
    @pytest.mark.parametrize("method", ["to_png", "to_multiqc", "show"])
    def test_methods_raise_before_compute(self, stub, tmp_path, method):
        args = (tmp_path / "out.png",) if method == "to_png" else ()

        with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
            getattr(stub, method)(*args)

    @pytest.mark.parametrize("as_str", [False, True])
    def test_to_png_writes_a_png_file_from_a_path_or_a_string(self, stub, tmp_path, as_str):
        stub.compute()
        out = tmp_path / "out.png"

        stub.to_png(str(out) if as_str else out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_as_a_base64_png_named_after_the_registry(self, stub):
        stub.compute()

        (section,) = stub.to_multiqc()

        assert (section.name, section.anchor) == ("stub_image", "stub_image")
        assert section.plot is None
        assert base64.b64decode(section.content.split("base64,", 1)[1].split('"', 1)[0]).startswith(PNG_MAGIC)

    def test_show_delegates_to_ipython_display(self, stub, displayed):
        stub.compute()

        stub.show()

        assert displayed == [stub._fig]


class TestPlotlyVisualization:
    def test_methods_raise_before_compute(self, tmp_path):
        for method, args in (("to_png", (tmp_path / "out.png",)), ("show", ())):
            with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
                getattr(_StubPlotlyPlot(), method)(*args)

    def test_to_png_writes_with_a_string_path_and_show_delegates_to_the_figure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("drevalpy.visualization.base.plotly_png_supported", lambda: True)
        plot = _StubPlotlyPlot()
        plot.compute()

        plot.to_png(tmp_path / "out.png")
        plot.show()

        assert plot._fig.written == [str(tmp_path / "out.png")]
        assert plot._fig.shown == 1

    def test_to_png_raises_where_plotly_cannot_render_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr("drevalpy.visualization.base.plotly_png_supported", lambda: False)
        plot = _StubPlotlyPlot()
        plot.compute()

        with pytest.raises(DrEvalError, match="not supported on Windows"):
            plot.to_png(tmp_path / "out.png")
        assert plot._fig.written == []
