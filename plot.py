# plot.py --- External plotting script: posterior joint probability distributions
#
# Design notes:
#   * It is independent from Analysis.py so it can be run standalone and its
#     style can be edited without touching the sampler;
#   * Parameter names, the total number of parameters (N_parameter) and the
#     bounds are all read from the `config` instance in Input.py;
#   * The plotting logic is self-contained (the SciPlotStyle dependency of the
#     original analysis code was removed);
#   * Styling follows the `engineering-figure-agent` publication rules: the
#     artwork is laid out at its final print size (one journal column per
#     parameter), the credible regions use two restrained, print-safe blues
#     with dark outlines, ticks are short and carry minor ticks, tick labels
#     are not rotated, and the figure is exported as PNG + vector PDF/SVG.
#
# Run inside universal_BA/ (HMC_sample.dat must exist first):
#   python3 plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator

from Input import config

# Times New Roman for all text (incl. the marginal distributions); STIX
# mathtext blends with Times for the LaTeX parameter labels ($f_n$, $f_l$).
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# Figure geometry: the artwork is laid out at its final print size, one
# journal column per parameter, so the font sizes below are the sizes the
# reader finally sees.  The whole figure is only ~6 in wide, so the text is
# deliberately generous -- 6-7 pt labels would be hard to read on screen and
# in print.  FS_ANNOT is the reference size: any future in-figure key/legend
# must use exactly FS_ANNOT, so it stays consistent with the $T_d$ annotation.
FS_LABEL = 14.0     # axis labels ($a_m$, $f_T$, ...)
FS_TICK = 10.5      # tick labels (the 95% CI bounds)
FS_ANNOT = 18.0     # figure annotation ($T_d = 180$ MeV)

plt.rcParams['axes.labelsize'] = FS_LABEL
plt.rcParams['xtick.labelsize'] = FS_TICK
plt.rcParams['ytick.labelsize'] = FS_TICK
plt.rcParams['axes.edgecolor'] = '#4D4D4D'      # neutral frame, not pure black
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.color'] = '#4D4D4D'
plt.rcParams['ytick.color'] = '#4D4D4D'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['ytick.major.size'] = 4.0
plt.rcParams['xtick.major.width'] = 0.9
plt.rcParams['ytick.major.width'] = 0.9
plt.rcParams['xtick.minor.size'] = 2.2
plt.rcParams['ytick.minor.size'] = 2.2
plt.rcParams['xtick.minor.width'] = 0.7
plt.rcParams['ytick.minor.width'] = 0.7
plt.rcParams['svg.fonttype'] = 'none'           # keep text editable in SVG
plt.rcParams['pdf.fonttype'] = 42               # embed TrueType in PDF
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

PANEL_WIDTH_IN = 1.44       # figure width/height per parameter, in inches
TICK_MINOR_DIV = 4          # minor ticks between two major ticks

# fraction of the 95% CI width padded on each side when adapting plot ranges
CI_PAD_FRAC = 0.5

# Contours drawn in the joint panels: exactly two, the 68% and 95% credible
# regions (seaborn's default is 10).  `levels` are iso-proportions of the mass
# *outside* the contour, so a p-credible region is requested as 1 - p; the
# trailing 1.0 closes the fill, otherwise only the 68-95% annulus would be
# shaded and the area inside the 68% contour would be left blank.
KDE_LEVELS = [1 - 0.95, 1 - 0.68, 1.0]
# the two credible-region boundaries only: contourf shades between levels but
# draws no line at them, so they are overlaid explicitly (the trailing fill
# level is dropped -- it sits at the density peak and would draw nothing)
KDE_LINE_LEVELS = KDE_LEVELS[:-1]

# Two restrained, print-safe blues: the outer band is the 95% region, the
# darker core is the 68% region.  Both are outlined in a dark blue so the two
# bands stay separable in greyscale printing.
FILL_COLORS = ['#BCD7EE', '#3775BA']
CI_LINE_COLOR = '#0F4D92'
CI_LINE_WIDTH = 1.2

# posterior median: black dash-dotted cross-hairs plus a central circle
MEDIAN_COLOR = 'k'
MEDIAN_LINESTYLE = '-.'
MEDIAN_LINE_WIDTH = 1.1
MEDIAN_MARKER = 'o'
MEDIAN_MARKER_SIZE = 4.5


class BayesianAnalysisPlotter:
    """Plot the posterior joint probability distributions of the parameters."""

    def __init__(self, data_path=None, bounds=None, central_values=None,
                 ci_95=None, ci_68=None):
        self.config = config
        self.data_path = data_path or config.sample_filepath
        self.all_samples = None
        self.bounds = bounds
        self.central_values = central_values
        self.ci_95 = ci_95
        self.ci_68 = ci_68
        self.labels = config.parameter_names
        self.N = config.N_parameter

    def load_data(self):
        """Load the first N_parameter columns of the sample file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Sample file does not exist: {self.data_path}")
        self.all_samples = np.loadtxt(self.data_path, usecols=range(self.N))
        print(f"Data loaded, shape: {self.all_samples.shape}")
        if self.bounds is None:
            if len(self.config.Bayesian_bound) == self.N:
                self.bounds = self.config.Bayesian_bound
            else:
                # fall back to the data range (+/-10%) if bounds are inconsistent
                self.bounds = []
                for i in range(self.all_samples.shape[1]):
                    lo, hi = np.min(self.all_samples[:, i]), np.max(self.all_samples[:, i])
                    rng = hi - lo
                    self.bounds.append((lo - 0.1 * rng, hi + 0.1 * rng))

    def plot_joint_distributions(self):
        """Draw all 2D joint distributions (lower triangle) and the marginal
        distributions (diagonal); the upper triangle is left blank."""
        if self.all_samples is None or self.all_samples.size == 0:
            self.load_data()
        if self.central_values is None:
            self.central_values = self.calculate_medians()
        if self.ci_95 is None:
            self.ci_95 = self.calculate_95ci()
        if self.ci_68 is None:
            self.ci_68 = self.calculate_68ci()

        n = self.all_samples.shape[1]
        # adaptive plot ranges: 95% CI expanded by CI_PAD_FRAC of its width on
        # each side (instead of showing the full prior box)
        plot_bounds = []
        for i in range(n):
            lo, hi = self.ci_95[0][i], self.ci_95[1][i]
            pad = CI_PAD_FRAC * (hi - lo)
            plot_bounds.append((lo - pad, hi + pad))
        # square corner plot, sized to the final print width
        side = PANEL_WIDTH_IN * n
        fig, axes = plt.subplots(n, n, figsize=(side, side))
        plt.subplots_adjust(hspace=0.12, wspace=0.12)

        label_size = plt.rcParams['axes.labelsize']
        for row in range(n):
            for col in range(n):
                ax = axes[row, col]

                if col < row:
                    # lower triangle: 2D joint distribution, shaded inside the
                    # 68% and 95% credible regions only (KDE_LEVELS)
                    sns.kdeplot(x=self.all_samples[:, col],
                                y=self.all_samples[:, row],
                                fill=True, bw_adjust=0.8, levels=KDE_LEVELS,
                                cmap=ListedColormap(FILL_COLORS),
                                ax=ax, zorder=1)
                    # overlay the two boundaries as dark, print-safe outlines
                    sns.kdeplot(x=self.all_samples[:, col],
                                y=self.all_samples[:, row],
                                fill=False, bw_adjust=0.8,
                                levels=KDE_LINE_LEVELS, color=CI_LINE_COLOR,
                                linewidths=CI_LINE_WIDTH, ax=ax, zorder=3)
                    ax.set_xlim(plot_bounds[col][0], plot_bounds[col][1])
                    ax.set_ylim(plot_bounds[row][0], plot_bounds[row][1])
                    ax.set_ylabel(self.labels[row] if col == 0 else '',
                                  fontsize=label_size, labelpad=4)
                    ax.set_xlabel(self.labels[col] if row == n - 1 else '',
                                  fontsize=label_size, labelpad=4)
                    # tick positions mark the 95% CI bounds (x from this
                    # column, y from this row); the y ticks must be pinned
                    # explicitly, otherwise the 95% CI labels would be attached
                    # to whatever ticks the auto-locator happens to pick
                    self._style_ticks(ax, (self.ci_95[0][col], self.ci_95[1][col]),
                                      (self.ci_95[0][row], self.ci_95[1][row]))
                    if col != 0:
                        ax.tick_params(labelleft=False)
                    else:
                        ax.set_yticklabels([f"{self.ci_95[0][row]:.2f}",
                                            f"{self.ci_95[1][row]:.2f}"])
                    if row != n - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xticklabels([f"{self.ci_95[0][col]:.2f}",
                                            f"{self.ci_95[1][col]:.2f}"])
                    # central (median) values: black dash-dotted guides + circle
                    self._median_guides(ax, self.central_values[col],
                                        self.central_values[row], both=True)

                elif col == row:
                    # diagonal: marginal distribution
                    sns.kdeplot(self.all_samples[:, row], fill=True,
                                color=FILL_COLORS[0], common_norm=True,
                                bw_adjust=0.8, ax=ax, linewidth=0, zorder=2)
                    sns.kdeplot(self.all_samples[:, row], fill=False,
                                color=CI_LINE_COLOR, common_norm=True,
                                bw_adjust=0.8, ax=ax,
                                linewidth=CI_LINE_WIDTH, zorder=3)
                    ax.set_xlim(plot_bounds[row][0], plot_bounds[row][1])
                    ax.set_yticks([])
                    ax.set_ylabel(self.labels[row] if row == 0 else '',
                                  fontsize=label_size, labelpad=4)
                    ax.set_xlabel(self.labels[row] if row == n - 1 else '',
                                  fontsize=label_size, labelpad=4)
                    self._style_ticks(ax, (self.ci_95[0][row], self.ci_95[1][row]))
                    if row == n - 1:
                        ax.set_xticklabels([f"{self.ci_95[0][row]:.2f}",
                                            f"{self.ci_95[1][row]:.2f}"])
                    else:
                        # avoid auto-formatted (non-2-decimal) numbers on the
                        # upper marginal panels; CI values are shown on the
                        # bottom row only
                        ax.tick_params(labelbottom=False)
                    self._median_guides(ax, self.central_values[row], None,
                                        both=False)

                else:
                    # upper triangle: empty
                    ax.axis('off')

        # align the left-column y labels deterministically: the joint panel
        # carries numeric y tick labels while the marginal panel does not,
        # so labelpad-based placement puts $f_l$ and $f_n$ at different
        # figure-x. Pin both labels to the same axes-fraction x coordinate
        # (align_ylabels proved unreliable here).
        for ax in axes[:, 0]:
            ax.yaxis.set_label_coords(-0.22, 0.5)

        # The whole upper-right block of the first row is blank; the figure
        # annotation (e.g. $T_d = 180$ MeV) is anchored to the right edge of
        # that block, where it stays clear of every panel.
        if config.figure_annotation:
            first, last = axes[0, 1].get_position(), axes[0, n - 1].get_position()
            fig.text(last.x1, first.y0 + 0.5 * first.height,
                     config.figure_annotation, fontsize=FS_ANNOT,
                     ha='right', va='center')

        os.makedirs(config.plot_output_dir, exist_ok=True)
        base = os.path.join(config.plot_output_dir, 'Joint_Distributions')
        # 600 dpi raster preview for slides + vector PDF/SVG for submission
        for suffix in ('.png', '.pdf', '.svg'):
            fig.savefig(base + suffix,
                        dpi=600 if suffix == '.png' else 300,
                        bbox_inches='tight', pad_inches=0.02)
            print(f"Joint distributions saved to {base}{suffix}")
        plt.close(fig)

    @staticmethod
    def _style_ticks(ax, x_bounds, y_bounds=None):
        """Mark the 95% CI bounds as major ticks and add minor ticks.

        The bounds are installed with set_xticks/set_yticks so the tick
        locations are fixed before any custom tick labels are attached.
        """
        ax.set_xticks(list(x_bounds))
        if y_bounds is not None:
            ax.set_yticks(list(y_bounds))
        ax.xaxis.set_minor_locator(AutoMinorLocator(TICK_MINOR_DIV))
        ax.yaxis.set_minor_locator(AutoMinorLocator(TICK_MINOR_DIV))

    @staticmethod
    def _median_guides(ax, x_value, y_value, both):
        """Black dash-dotted guides through the posterior median, plus a
        central circle marker on the joint panels."""
        ax.axvline(x_value, color=MEDIAN_COLOR, ls=MEDIAN_LINESTYLE,
                   lw=MEDIAN_LINE_WIDTH, zorder=2)
        if both:
            ax.axhline(y_value, color=MEDIAN_COLOR, ls=MEDIAN_LINESTYLE,
                       lw=MEDIAN_LINE_WIDTH, zorder=2)
            ax.plot(x_value, y_value, marker=MEDIAN_MARKER,
                    ms=MEDIAN_MARKER_SIZE, color=MEDIAN_COLOR, zorder=4)

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def calculate_medians(self):
        if self.all_samples is None or self.all_samples.size == 0:
            self.load_data()
        medians = np.median(self.all_samples, axis=0)
        for name, m in zip(self.labels, medians):
            print(f"Median of {name}: {m:.4f}")
        return medians

    def calculate_95ci(self):
        if self.all_samples is None or self.all_samples.size == 0:
            self.load_data()
        lower = np.percentile(self.all_samples, 2.5, axis=0)
        upper = np.percentile(self.all_samples, 97.5, axis=0)
        for name, lo, hi in zip(self.labels, lower, upper):
            print(f"95% CI of {name}: [{lo:.4f}, {hi:.4f}]")
        return lower, upper

    def calculate_68ci(self):
        if self.all_samples is None or self.all_samples.size == 0:
            self.load_data()
        lower = np.percentile(self.all_samples, 16, axis=0)
        upper = np.percentile(self.all_samples, 84, axis=0)
        for name, lo, hi in zip(self.labels, lower, upper):
            print(f"68% CI of {name}: [{lo:.4f}, {hi:.4f}]")
        return lower, upper


if __name__ == "__main__":
    config.validate()
    plotter = BayesianAnalysisPlotter()
    plotter.plot_joint_distributions()
