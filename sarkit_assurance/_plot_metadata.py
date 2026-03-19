"""Utilities for generating plots of a file's metadata"""

import html
import itertools
import pathlib
import webbrowser

import plotly.offline.offline

from . import names

UP_ARROW = "\N{UPWARDS ARROW WITH EQUILATERAL ARROWHEAD}"


# Javascript that will link selectedpoints across Plotly traces.
# Each trace to link should have the same number of ordered points and have their meta property set to 'link'.
PLOTLY_POST_SCRIPT = """
    var graph_div = document.getElementById("{plot_id}");
    graph_div.on('plotly_selected', function(eventData) {
        if (eventData == null) {
            Plotly.restyle(graph_div, {selectedpoints: [null]})
        } else {
            let point_numbers_single = [];
            eventData.points.forEach(pt => {
                if (pt.data.meta === 'link') {
                    point_numbers_single.push(pt.pointNumber)
                }
            });
            let point_numbers =[];
            let plot_numbers = [];
            graph_div.data.forEach((item, ndx) => {
                if (item.meta === 'link') {
                    point_numbers.push(point_numbers_single)
                    plot_numbers.push(ndx);
                }
            });
            Plotly.restyle(graph_div, {selectedpoints: point_numbers}, plot_numbers);
        }
    });
    Plotly.restyle(graph_div, {'unselected': {marker: {opacity: 0.02}}});
"""


class Plotter:
    """A metadata plotter class.

    Provides `plot_*` methods that generate figures of various aspects of a file's metadata.
    A `plot_` method shall:

        * Always return a list of figures. If a plot is unavailable, the list shall be empty.
        * Populate each figures' `meta` layout attribute with a unique string identifier.

    """

    def __init__(self, title):
        self.title = title
        self.plotters = [
            attr
            for name in dir(self)
            if name.startswith("plot_")
            and hasattr((attr := getattr(self, name)), "__call__")
        ]

    @staticmethod
    def titlefy_plotter(plotter):
        return plotter.removeprefix("plot_").replace("_", " ").title()

    @staticmethod
    def get_plotly_js():
        return f'<script type="text/javascript">{plotly.offline.offline.get_plotlyjs()}</script>'

    def format_title(self, raw):
        return f"<b>{html.escape(raw)}</b> - <i>{self.title}</i><br>"

    def make_available_figures(self, plotter_names=None):
        """Returns a dict mapping `plot_` names to their list of generated figures for available plotters."""
        if plotter_names is None:
            plotter_names = [x.__name__.removeprefix("plot_") for x in self.plotters]
        return {
            func.__name__: figs
            for func in self.plotters
            if func.__name__.removeprefix("plot_") in plotter_names and (figs := func())
        }

    def save_separate(self, output_dir, prefix, figs=None, auto_open=False):
        """Save figures to separate HTML files.

        Args
        ----
        output_dir: path-like
            Directory where generated HTML files will be written
        prefix: str
            Prefix used in output filenames
        figs: dict
            Dict mapping plot function names to their list of generated figures.
            If None, the output of `make_available_figures` is used.
        auto_open: bool
            If ``True``, open figures after saving.

        """
        figs = self.make_available_figures() if figs is None else figs
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fig in itertools.chain(*figs.values()):
            sanitized_stem = names.sanitize_name(fig["layout"]["meta"])
            fig.write_html(
                str(output_dir / f"{prefix}{sanitized_stem}.html"),
                auto_open=auto_open,
                post_script=PLOTLY_POST_SCRIPT,
            )

    def make_plot_divs(self, figs=None):
        """Make specified figures into HTML divs

        Args
        ----
        figs: dict
            Dict mapping plot function names to their list of generated figures.
            If None, the output of `make_available_figures` is used.

        Returns
        -------
        divs_by_id: dict
            Dict keyed by plotter names.  Values are an html div containing a header
            with an id of the key and another div containing the plotly figure.

        """
        figs = self.make_available_figures() if figs is None else figs

        divs_by_id = dict()

        # Add individual plots
        for plotter, figs in figs.items():
            plotter_html = ["<div>"]
            plotter_html.append(
                f'<h2 id="{plotter}">{self.titlefy_plotter(plotter)}'
                f' <a href="#top">&#{ord(UP_ARROW)}</a></h2>'
            )
            for fig in figs:
                plotter_html.append(
                    fig.to_html(
                        full_html=False,
                        include_plotlyjs=False,
                        post_script=PLOTLY_POST_SCRIPT,
                        default_width=1280,
                        default_height=800,
                    )
                )
            plotter_html.append("</div>")
            divs_by_id[plotter] = "".join(plotter_html)

        return divs_by_id

    def save_combined(self, output_dir, prefix, figs=None, auto_open=False):
        """Save figures to a combined HTML file.

        Args
        ----
        output_dir: path-like
            Directory where generated HTML files will be written
        prefix: str
            Prefix used in output filenames
        figs: dict
            Dict mapping plot function names to their list of generated figures.
            If None, the output of `make_available_figures` is used.
        auto_open: bool
            If ``True``, open figures after saving.

        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_html = pathlib.Path(output_dir) / f"{prefix}metadata.html"
        output_html.unlink(missing_ok=True)

        # Make plotter divs
        figs = self.make_available_figures() if figs is None else figs
        divs_by_id = self.make_plot_divs(figs)

        # Build html
        header = '<html>\n<head><meta charset="utf-8"/></head>\n<body>\n'

        # Add table of contents
        toc = (
            "<div>"
            "<h1>Contents</h1>"
            "<ul>"
            + "".join(
                f'<li><a href="#{p}">{self.titlefy_plotter(p)}</a></li>'
                for p in divs_by_id
            )
            + "</ul>"
            "</div>"
        )
        html_txt_segments = [header]
        html_txt_segments.append(self.get_plotly_js())
        html_txt_segments.append(toc)
        html_txt_segments.extend(divs_by_id.values())
        html_txt_segments.append("</body>\n</html>")
        output_html.write_text("".join(html_txt_segments))

        if auto_open:
            webbrowser.open(f"file://{output_html.resolve()}")
