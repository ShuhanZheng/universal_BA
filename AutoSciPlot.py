"""
AutoSciPlot -- Automated Scientific Plotting Wrapper

Usage modes:
  1. Import:  from AutoSciPlot import AutoPlot
  2. Script:  edit the CONFIG dict at the bottom, then run `python AutoSciPlot.py`
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import SciPlotStyle as SciPS

# ---------------------------------------------------------------------------
# Default color cycle: purple, orange, green, black, red
# ---------------------------------------------------------------------------
COLOR_PALETTE = ["#8A2BE2", "#FF8C00", "#2E8B57", "#000000", "#DC143C"]
COLOR_NAMES = ["purple", "orange", "green", "black", "red"]


class AutoPlot:
    """Build a publication-quality plot from data files with minimal boilerplate."""

    def __init__(self):
        self._datasets = []
        self._arrays = []
        self._hists = []
        self._xlabel = ""
        self._ylabel = ""
        self._xlim = None
        self._ylim = None
        self._xscale = "linear"
        self._yscale = "linear"
        self._legend_loc = "best"
        self._legend_frameon = False
        self._legend_ncol = None
        self._title = ""
        self._figsize = None
        self._make_square = True
        self._x_major_tick = None
        self._x_minor_tick = None
        self._y_major_tick = None
        self._y_minor_tick = None
        self._texts = []

    # ---- fluent config API ------------------------------------------------

    def add_data(self, filepath, x_col=0, y_col=1, label=None, color=None,
                 skiprows=0, delimiter=None,
                 x_err_col=None, y_err_col=None, **plot_kwargs):
        """Register one curve from a text data file.

        Parameters
        ----------
        filepath : str
            Path to the data file (space/tab/comma-separated columns).
        x_col, y_col : int
            Zero-based column indices.
        label : str or None
            Legend label for this curve.
        color : str or None
            Override the automatic color assignment.
        skiprows : int
            Number of header rows to skip.
        delimiter : str or None
            Column delimiter (None = any whitespace).
        x_err_col, y_err_col : int or (int, int) or None
            Error-bar columns.  If ``None``, the curve is drawn with ``ax.plot``.
            Otherwise the curve is drawn with ``ax.errorbar``:
              - int  : symmetric error, magnitude taken from that column
              - pair ``(lo, hi)`` : asymmetric error; the two columns give the
                lower and upper deviations, and their absolute values are used
                (so signed ``unc_down`` / ``unc_up`` columns work directly).
        plot_kwargs
            Passed through to ``ax.plot()`` or ``ax.errorbar()``
            (e.g. fmt, linestyle, marker, linewidth, markersize, capsize, elinewidth).
        """
        self._datasets.append({
            "filepath": filepath,
            "x_col": x_col, "y_col": y_col,
            "label": label, "color": color,
            "skiprows": skiprows, "delimiter": delimiter,
            "x_err_col": x_err_col, "y_err_col": y_err_col,
            "plot_kwargs": plot_kwargs,
        })
        return self

    def add_array(self, x, y, label=None, color=None, **plot_kwargs):
        """Register one curve from in-memory arrays instead of a data file.

        Unlike add_data (which cycles through COLOR_PALETTE), a None color
        here leaves the choice to matplotlib's default cycle, which keeps
        many overlapping quick-look curves (e.g. Markov chains) mutually
        distinguishable; pass color explicitly for palette-consistent
        styling.  plot_kwargs are forwarded to ``ax.plot()``.
        """
        self._arrays.append({
            "x": np.asarray(x), "y": np.asarray(y),
            "label": label, "color": color,
            "plot_kwargs": plot_kwargs,
        })
        return self

    def add_hist(self, values, bins=30, density=True, label=None, color=None,
                 **hist_kwargs):
        """Register a histogram of in-memory values.

        Histograms are drawn first, underneath every curve registered with
        add_data / add_array, so a reference distribution (e.g. a Gaussian
        probability density) can be overlaid on top of the bars.

        Parameters
        ----------
        values : array-like
            Sample to be binned (flattened before use).
        bins : int or sequence
            Passed to ``ax.hist``; an explicit sequence of edges is useful to
            make several panels share the same binning.
        density : bool
            Normalize the histogram to a probability density (default True).
        label : str or None
            Legend label for this histogram.
        color : str or None
            Override the automatic color assignment; the bar outline follows
            ``edgecolor`` in hist_kwargs.
        hist_kwargs
            Passed through to ``ax.hist()`` (e.g. alpha, edgecolor, linewidth,
            histtype, range).
        """
        self._hists.append({
            "values": np.asarray(values).ravel(),
            "bins": bins, "density": density,
            "label": label, "color": color,
            "hist_kwargs": hist_kwargs,
        })
        return self

    def set_labels(self, xlabel="", ylabel=""):
        """Set axis labels (supports LaTeX math mode, e.g. r\"$p_T$ (GeV)\")."""
        self._xlabel = xlabel
        self._ylabel = ylabel
        return self

    def set_limits(self, xlim=None, ylim=None):
        """Set both axis ranges at once.  ``None`` means auto-detect."""
        self._xlim = xlim
        self._ylim = ylim
        return self

    def set_xlim(self, xmin=None, xmax=None):
        """Set x-axis range.  ``None`` means auto-detect."""
        self._xlim = (xmin, xmax)
        return self

    def set_ylim(self, ymin=None, ymax=None):
        """Set y-axis range.  ``None`` means auto-detect."""
        self._ylim = (ymin, ymax)
        return self

    def set_scales(self, xscale="linear", yscale="linear"):
        """Set axis scales ('linear' or 'log')."""
        self._xscale = xscale
        self._yscale = yscale
        return self

    def set_ticks(self, x_major=None, x_minor=None, y_major=None, y_minor=None):
        """Set major and minor tick spacing for each axis.

        If *only* major is given, minor defaults to major / 4.
        ``None`` (default) keeps SciPlotStyle's AutoMinorLocator.
        """
        self._x_major_tick = x_major
        self._x_minor_tick = x_minor
        self._y_major_tick = y_major
        self._y_minor_tick = y_minor
        return self

    def set_legend(self, loc="best", frameon=False, ncol=None):
        """Configure legend position, frame visibility and column count."""
        self._legend_loc = loc
        self._legend_frameon = frameon
        self._legend_ncol = ncol
        return self

    def set_title(self, title):
        """Set an optional figure title."""
        self._title = title
        return self

    def set_figsize(self, width, height):
        """Override the default figure size (inches)."""
        self._figsize = (width, height)
        return self

    def set_square(self, make_square):
        """Toggle SciPlotStyle's square-axes enforcement (default True).

        Set False for wide diagnostic figures (e.g. Markov-chain traces)."""
        self._make_square = make_square
        return self

    def text(self, x, y, s, **kwargs):
        """Add a text annotation to the axes.

        Parameters are forwarded to ``ax.text()``, e.g.
        ``.text(0.5, 0.95, r'$...$', ha='center', va='top')``.
        """
        self._texts.append({"x": x, "y": y, "s": s, "kwargs": kwargs})
        return self

    # ---- tick helpers -----------------------------------------------------

    def _apply_ticks(self, ax):
        if self._x_major_tick is not None:
            ax.xaxis.set_major_locator(MultipleLocator(self._x_major_tick))
            minor = (self._x_minor_tick if self._x_minor_tick is not None
                     else self._x_major_tick / 4)
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
        elif self._x_minor_tick is not None:
            ax.xaxis.set_minor_locator(MultipleLocator(self._x_minor_tick))

        if self._y_major_tick is not None:
            ax.yaxis.set_major_locator(MultipleLocator(self._y_major_tick))
            minor = (self._y_minor_tick if self._y_minor_tick is not None
                     else self._y_major_tick / 4)
            ax.yaxis.set_minor_locator(MultipleLocator(minor))
        elif self._y_minor_tick is not None:
            ax.yaxis.set_minor_locator(MultipleLocator(self._y_minor_tick))

    # ---- main entry point -------------------------------------------------

    @staticmethod
    def _resolve_err(data, spec):
        """Convert an error-column spec into a matplotlib err array.

        ``int`` -> symmetric magnitude from that column;
        ``(lo, hi)`` -> asymmetric [lower, upper] magnitudes (abs taken).
        """
        if spec is None:
            return None
        if isinstance(spec, (tuple, list)):
            lo, hi = spec
            return [np.abs(data[:, lo]), np.abs(data[:, hi])]
        return np.abs(data[:, spec])

    def plot(self, save_path=None, show=True, ax=None):
        """Generate the figure and optionally save / display it.

        Parameters
        ----------
        save_path : str or None
            Where to write the figure (300 dpi, tight bounding box).
        show : bool
            Call ``plt.show()`` at the end.
        ax : matplotlib Axes or None
            Draw into an existing axes instead of creating a new figure; used
            to build multi-panel figures (one AutoPlot per panel).  The caller
            then owns the figure layout, so SciPlotStyle's tight_layout /
            subplots_adjust pass is skipped for that axes.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        external_ax = ax is not None
        if external_ax:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=self._figsize)

        # histograms first: they are the background of the panel
        for i, spec in enumerate(self._hists):
            kw = spec["hist_kwargs"].copy()
            kw.pop("color", None)
            kw.pop("label", None)
            ax.hist(spec["values"], bins=spec["bins"], density=spec["density"],
                    color=spec["color"] or COLOR_PALETTE[i % len(COLOR_PALETTE)],
                    label=spec["label"], zorder=1, **kw)

        for i, ds in enumerate(self._datasets):
            data = np.loadtxt(ds["filepath"], skiprows=ds["skiprows"],
                              delimiter=ds["delimiter"])
            x = data[:, ds["x_col"]]
            y = data[:, ds["y_col"]]

            color = ds["color"] or COLOR_PALETTE[i % len(COLOR_PALETTE)]
            kw = ds["plot_kwargs"].copy()
            kw.pop("color", None)

            xerr = self._resolve_err(data, ds["x_err_col"])
            yerr = self._resolve_err(data, ds["y_err_col"])

            if xerr is not None or yerr is not None:
                kw.setdefault("fmt", "o")   # marker for errorbar plots
                ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                            color=color, label=ds["label"], **kw)
            else:
                ax.plot(x, y, color=color, label=ds["label"], **kw)

        for spec in self._arrays:
            kw = spec["plot_kwargs"].copy()
            if spec["color"] is not None:
                kw["color"] = spec["color"]
            ax.plot(spec["x"], spec["y"], label=spec["label"], **kw)

        ax = SciPS.set_plot_style(ax, make_square=self._make_square,
                                  reduce_margins=not external_ax)

        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        ax.set_xscale(self._xscale)
        ax.set_yscale(self._yscale)

        if self._xlim is not None:
            ax.set_xlim(self._xlim)
        if self._ylim is not None:
            ax.set_ylim(self._ylim)

        self._apply_ticks(ax)
        if self._title:
            ax.set_title(self._title)

        has_labels = (any(ds["label"] is not None for ds in self._datasets)
                      or any(spec["label"] is not None for spec in self._arrays)
                      or any(spec["label"] is not None for spec in self._hists))
        if has_labels:
            legend_kw = {"loc": self._legend_loc, "frameon": self._legend_frameon}
            if self._legend_ncol is not None:
                legend_kw["ncol"] = self._legend_ncol
            ax.legend(**legend_kw)

        for t in self._texts:
            ax.text(t["x"], t["y"], t["s"], **t["kwargs"])

        if save_path:
            # fig.savefig, not plt.savefig: with an external axes the current
            # figure need not be the one that was drawn into
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig, ax


# ===========================================================================
# Standalone script mode: edit the CONFIG dict below, then run this file.
# ===========================================================================
if __name__ == "__main__":
    # -- user configuration ------------------------------------------------
    CONFIG = {
        # One entry per curve.  Colors are auto-assigned in palette order
        # (purple → orange → green → black → red); override with "color".
        "data": [
            {"file": "data1.dat", "label": "Curve 1"},
            {"file": "data2.dat", "label": "Curve 2"},
        ],

        "xlabel": r"$x$",
        "ylabel": r"$y$",

        "xlim": None,            # None = auto; or (xmin, xmax)
        "ylim": None,
        "xscale": "linear",
        "yscale": "linear",

        "legend_loc": "best",
        "legend_frameon": False,

        "title": "",
        "save_path": None,       # e.g. "output.pdf"; None = don't save
    }

    # -- build & plot ------------------------------------------------------
    plotter = AutoPlot()

    for d in CONFIG["data"]:
        plotter.add_data(
            filepath=d["file"],
            label=d.get("label"),
            color=d.get("color"),
            x_col=d.get("x_col", 0),
            y_col=d.get("y_col", 1),
            skiprows=d.get("skiprows", 0),
        )

    plotter.set_labels(CONFIG["xlabel"], CONFIG["ylabel"])
    plotter.set_limits(CONFIG["xlim"], CONFIG["ylim"])
    plotter.set_scales(CONFIG["xscale"], CONFIG["yscale"])
    plotter.set_legend(CONFIG["legend_loc"], CONFIG["legend_frameon"])

    if CONFIG["title"]:
        plotter.set_title(CONFIG["title"])

    plotter.plot(save_path=CONFIG.get("save_path"))
