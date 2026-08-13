"""Tests for :mod:`drevalpy.visualization.base`.

``ImageVisualization`` supplies ``to_png``/``to_multiqc``/``show`` to every
matplotlib-backed plot, so its three ``RuntimeError`` guards and the base64
embedding are covered here once via a stub subclass rather than repeatedly in
each plot's own module.
"""

from __future__ import annotations

import base64
import sys
import types

import matplotlib.pyplot as plt
import pytest

from drevalpy.visualization.base import ImageVisualization, Section, Visualization

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _StubImagePlot(ImageVisualization):
    """Smallest possible ImageVisualization: a one-line figure."""

    registry_name = "stub_image"

    def compute(self, result=None, dataset=None) -> None:
        self._fig = self._create_figure()

    def _create_figure(self):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.plot([0, 1], [0, 1])
        return fig


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def stub() -> _StubImagePlot:
    return _StubImagePlot()


@pytest.fixture
def displayed(monkeypatch: pytest.MonkeyPatch) -> list:
    """Install a recording ``IPython.display.display`` and return what it was handed.

    The stub replaces every real ``IPython*`` entry rather than only filling gaps:
    IPython 9 is a real dependency of the dev environment, so leaving it reachable
    would make the result depend on whether an earlier test imported it.

    It also has to look enough like IPython for matplotlib, which reads
    ``sys.modules["IPython"].version_info`` (and ``get_ipython()``) once per process
    the first time a canvas is created. A bare ``ModuleType`` made that one-off probe
    raise ``AttributeError`` whenever this test happened to draw the first figure in
    the process - which is why it passed serially and failed under ``-n auto``.
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


class TestSection:
    def test_optional_fields_default_to_empty(self):
        section = Section(name="Some plot", anchor="some_plot")

        assert (section.description, section.plot, section.content) == ("", None, None)

    def test_carries_a_native_plot_object(self):
        sentinel = object()

        section = Section(name="n", anchor="a", description="d", plot=sentinel)

        assert section.plot is sentinel

    def test_equality_is_by_value(self):
        assert Section(name="n", anchor="a") == Section(name="n", anchor="a")


class TestVisualizationContract:
    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Visualization()  # type: ignore[abstract]

    def test_declares_the_full_abstract_surface(self):
        assert Visualization.__abstractmethods__ == frozenset({"compute", "to_png", "to_multiqc", "show"})

    def test_registry_name_defaults_to_empty(self):
        assert Visualization.registry_name == ""

    def test_image_visualization_adds_only_create_figure(self):
        assert ImageVisualization.__abstractmethods__ == frozenset({"compute", "_create_figure"})


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, stub, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            stub.to_png(tmp_path / "out.png")

    def test_to_multiqc_raises(self, stub):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            stub.to_multiqc()

    def test_show_raises(self, stub):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            stub.show()


class TestImageVisualizationRendering:
    def test_to_png_writes_a_png_file(self, stub, tmp_path):
        stub.compute()
        out = tmp_path / "out.png"

        stub.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_to_png_accepts_a_string_path(self, stub, tmp_path):
        stub.compute()
        out = tmp_path / "str.png"

        stub.to_png(str(out))

        assert out.exists()

    def test_to_multiqc_returns_one_section_named_after_the_registry(self, stub):
        stub.compute()

        sections = stub.to_multiqc()

        assert [(s.name, s.anchor) for s in sections] == [("stub_image", "stub_image")]

    def test_to_multiqc_embeds_the_figure_as_base64_png(self, stub):
        stub.compute()

        content = stub.to_multiqc()[0].content

        assert content is not None
        payload = content.split("base64,", 1)[1].split('"', 1)[0]
        assert base64.b64decode(payload).startswith(PNG_MAGIC)

    def test_to_multiqc_section_carries_no_native_plot(self, stub):
        stub.compute()

        assert stub.to_multiqc()[0].plot is None

    def test_show_delegates_to_ipython_display(self, stub, displayed):
        stub.compute()

        stub.show()

        assert displayed == [stub._fig]
