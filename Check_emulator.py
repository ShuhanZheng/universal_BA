# Check_emulator.py --- Emulator validation: normalized-residual histograms
#
# Independent correctness check of the PCA + GP emulator that Analysis.py
# samples with. The script re-trains the emulator on subsets of the design
# table, predicts the model output at the points of the subset it has NOT seen,
# and tests whether the GP predictive uncertainty is correctly calibrated.
#
# The diagnostic follows J. E. Bernhard, "Bayesian parameter estimation for
# relativistic heavy-ion collisions", arXiv:1804.06469, section 4.3.4
# "Validation", figure 4.9 (center panel). A GP predicts a probability
# distribution rather than a single number, so the meaningful question is not
# "does the emulator reproduce the calculation?" but "are the calculations
# distributed the way the emulator says they are?". For every principal
# component m and every held-out point n, the normalized residual
#
#       z_{m,n} = ( P_m^pred(x_n) - P_m^true(x_n) ) / sigma_{P,m}(x_n)
#
# must therefore follow a standard normal distribution, N(0, 1), with
#
#       P_m^pred(x_n)    = gp_m.predict(x_n) mean          (PC-space prediction)
#       P_m^true(x_n)    = m-th PC of the held-out theory row, i.e.
#                          (y_th(x_n) - pca.mean_) @ pca.components_[m]
#       sigma_{P,m}(x_n) = gp_m.predict(x_n) std           (PC-space uncertainty)
#
# sigma_{P,m} is exactly the gp.predict(..., return_std=True) that feeds
# Sigma_th(x) (README Eq. 7) and hence the likelihood, so the histogram tests
# the very uncertainty the sampler relies on, not an idealized one. It is the
# predictive std of a *noisy* observation, i.e. it includes the white-noise term
# of the kernel as well as the interpolation uncertainty -- consistent with the
# GP's own model of the training table.
#
# The 5 retained principal components are histogrammed separately, because the
# emulator quality is not uniform across them: the leading components carry the
# smooth parameter dependence, while the trailing ones are closer to noise and
# their hyperparameters are much less constrained.
#
# No separate validation sample exists in this repository, so the held-out
# points are carved out of the design table:
#   * N_FOLDS = k > 1 : k-fold cross-validation; every design point is tested
#                       exactly once, by an emulator that never saw it, and the
#                       residuals of all folds are pooled (default);
#   * N_FOLDS = 1     : a single train/test split of size TEST_FRACTION.
# Each fold's emulator is built by Interpolator.fit, i.e. the production PCA+GP
# code path with the production kernel and hyperparameter optimization, so this
# check cannot silently drift away from what Analysis.py actually samples with.
# The price is that the fold emulators are trained on fewer points than the
# deployed one (e.g. 160 instead of 200), which makes the test conservative.
#
# Run inside universal_BA/:
#   python3 Check_emulator.py
# Output: a per-component calibration table on the terminal and
# <plot_output_dir>/Emulator_Normalized_Residuals.{png,pdf}.

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from sklearn.model_selection import KFold

from Input import config
from Class import Interpolator
# NB: AutoSciPlot imports SciPlotStyle, which installs its own rcParams at
# import time; the style block below therefore has to come after this import.
from AutoSciPlot import AutoPlot

# ---------------------------------------------------------------------------
# Validation settings (everything else is taken from Input.py)
# ---------------------------------------------------------------------------
N_FOLDS = 5             # k > 1: k-fold cross-validation; 1: single train/test split
TEST_FRACTION = 0.25    # size of the test set, only used when N_FOLDS == 1
RANDOM_SEED = 42        # seed of the shuffle / split (reproducibility)
N_BINS = 20             # bins per histogram (shared by all panels)

# ---------------------------------------------------------------------------
# Figure style: a wide multi-panel strip, so the font and tick sizes of
# SciPlotStyle (tuned for a large square single panel) are scaled down here.
# The colors are those of the joint-distribution figure (plot.py): light blue
# bars with a dark blue outline, black reference curve.
# ---------------------------------------------------------------------------
PANEL_WIDTH_IN = 1.55   # figure width per principal component
PANEL_HEIGHT_IN = 2.15
FS_LABEL = 11.0         # axes labels
FS_TICK = 9.0
FS_TITLE = 10.0
FS_ANNOT = 7.8          # per-panel annotations (mean/std, N(0,1) key)

HIST_COLOR = '#BCD7EE'
HIST_EDGE = '#0F4D92'
HIST_LINE_WIDTH = 0.7
GAUSS_COLOR = 'k'
GAUSS_LINE_WIDTH = 1.6
ANNOT_COLOR = '#4D4D4D'     # neutral gray, as the axis frame: the black text is
                            # then unambiguously the key of the black curve

YMAX_HEADROOM = 1.35    # y range = headroom * (tallest bar or N(0,1) peak);
                        # the resulting empty top strip holds the annotations
XMIN_LIMIT = 3.5        # smallest half-range of the z axis (widened if needed)
ZPLOT_LIMIT = 6.0       # largest half-range plotted: a single extreme residual
                        # would otherwise squeeze all panels into a spike; the
                        # points beyond it are still counted in the table and
                        # named in the panel annotation

plt.rcParams.update({
    'font.size': FS_TICK,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.minor.size': 2.2,
    'ytick.minor.size': 2.2,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.minor.width': 0.7,
    'ytick.minor.width': 0.7,
    'axes.edgecolor': '#4D4D4D',    # neutral frame, not pure black
    'axes.linewidth': 1.0,
    'xtick.color': '#4D4D4D',
    'ytick.color': '#4D4D4D',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'svg.fonttype': 'none',         # keep text editable in SVG
    'pdf.fonttype': 42,             # embed TrueType in PDF
})


class EmulatorValidator:
    """Validate the PCA + GP emulator with normalized residuals.

    The emulator is rebuilt from scratch in every fold (PCA and GPs together),
    so a residual is always produced by an emulator that was neither trained on
    that point nor allowed to see it when the PCA basis was constructed.
    """

    def __init__(self, config, n_folds=N_FOLDS, test_fraction=TEST_FRACTION,
                 random_seed=RANDOM_SEED):
        if config.analytical:
            raise ValueError(
                "Check_emulator.py validates the PCA+GP emulator, which exists "
                "for analytical=False only. The analytical branch has no "
                "predictive uncertainty to test (Sigma_th = 0).")
        self.config = config
        self.n_folds = int(n_folds)
        self.test_fraction = float(test_fraction)
        self.random_seed = int(random_seed)
        self.parameters = None          # design table {x_n}
        self.theory = None              # theory table {y_th(x_n)}
        self.residuals = None           # (n_pc, n_test) normalized residuals
        self.explained_variance_ratio = None   # (n_folds, n_pc)
        self.n_test_per_fold = []       # number of held-out points of each fold
        self._stats = {}                # cached statistics(), keyed by component

    # ------------------------------------------------------------------
    # data and folds
    # ------------------------------------------------------------------
    def load_data(self):
        """Load the design and theory tables from Input.py's file names."""
        self.parameters = np.loadtxt(
            os.path.join(self.config.input_dir, self.config.parameter_file))
        self.theory = np.loadtxt(
            os.path.join(self.config.input_dir, self.config.theoretical_data_file))
        print(f"Design table {{x_n}}:          {self.parameters.shape}")
        print(f"Theory table {{y_th(x_n)}}:    {self.theory.shape}")
        return self

    def _iter_splits(self, n_points):
        """Yield (train_index, test_index) of every fold."""
        if self.n_folds > 1:
            kfold = KFold(n_splits=self.n_folds, shuffle=True,
                          random_state=self.random_seed)
            yield from kfold.split(np.arange(n_points))
        else:
            rng = np.random.default_rng(self.random_seed)
            shuffled = rng.permutation(n_points)
            n_test = max(1, int(round(self.test_fraction * n_points)))
            yield shuffled[n_test:], shuffled[:n_test]

    # ------------------------------------------------------------------
    # the check itself
    # ------------------------------------------------------------------
    def run(self):
        """Train one emulator per fold and collect the normalized residuals."""
        if self.parameters is None:
            self.load_data()
        n_points = self.parameters.shape[0]
        n_pc = self.config.n_pca_components
        residual_blocks = []
        ratios = []

        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(self._iter_splits(n_points), start=1):
            print("\n" + "-" * 70)
            print(f"Fold {fold}/{self.n_folds}: training on {len(train_idx)} points, "
                  f"testing on {len(test_idx)} points")
            print("-" * 70)
            interpolator = Interpolator(self.config)
            interpolator.fit(self.parameters[train_idx], self.theory[train_idx])

            # PC-space truth: the held-out theory rows projected onto the PCA
            # basis of this fold's training set
            pca_true = interpolator.pca.transform(self.theory[test_idx])

            pca_pred = np.empty_like(pca_true)
            pca_std = np.empty_like(pca_true)
            for m, gp in enumerate(interpolator.gpr_model):
                mean, std = gp.predict(self.parameters[test_idx], return_std=True)
                pca_pred[:, m] = np.ravel(mean)
                pca_std[:, m] = np.ravel(std)

            z = (pca_pred - pca_true) / pca_std          # (n_test, n_pc)
            residual_blocks.append(z.T)                  # -> (n_pc, n_test)
            ratios.append(interpolator.pca.explained_variance_ratio_)
            self.n_test_per_fold.append(len(test_idx))
            print(f"Fold {fold}: {len(test_idx)} held-out points, "
                  f"median |z| = {np.median(np.abs(z)):.3f} (ideal 0.674), "
                  f"std(z) = {np.std(z):.3f} (ideal 1.000)")

        self.residuals = np.hstack(residual_blocks)
        self.explained_variance_ratio = np.array(ratios)
        print(f"\nValidation finished in {time.time() - start:.1f} s: "
              f"{self.residuals.shape[1]} held-out predictions per principal "
              f"component.")
        if self.residuals.shape[0] != n_pc:      # cannot happen, but be explicit
            raise RuntimeError("Residual array does not match n_pca_components.")
        return self.residuals

    # ------------------------------------------------------------------
    # numbers
    # ------------------------------------------------------------------
    def statistics(self, m):
        """Calibration statistics of principal component m against N(0, 1)."""
        if m in self._stats:
            return self._stats[m]
        z = self.residuals[m]
        ks_stat, ks_p = kstest(z, 'norm')
        self._stats[m] = {
            "n": z.size,
            "mean": np.mean(z),
            "std": np.std(z, ddof=1),
            "cov1": np.mean(np.abs(z) < 1.0),
            "cov2": np.mean(np.abs(z) < 2.0),
            "max": np.abs(z).max(),
            "ks_stat": ks_stat,
            "ks_p": ks_p,
        }
        return self._stats[m]

    def report(self):
        """Print the per-component calibration table."""
        if self.residuals is None:
            raise ValueError("No residuals yet; run() first.")
        print("\n========== Emulator validation: normalized residuals ==========")
        print("Ideal values: mean 0, std 1, |z|<1 -> 0.683, |z|<2 -> 0.954, "
              "KS p-value uniform")
        print(f"{'PC':>3} {'N':>5} {'mean':>8} {'std':>8} "
              f"{'|z|<1':>8} {'|z|<2':>8} {'max|z|':>7} {'KS stat':>8} {'KS p':>8}  "
              f"{'variance':>9}")
        for m in range(self.residuals.shape[0]):
            s = self.statistics(m)
            evr = 100 * np.mean(self.explained_variance_ratio[:, m])
            print(f"{m + 1:>3} {s['n']:>5} {s['mean']:>8.3f} {s['std']:>8.3f} "
                  f"{s['cov1']:>8.3f} {s['cov2']:>8.3f} {s['max']:>7.2f} "
                  f"{s['ks_stat']:>8.3f} {s['ks_p']:>8.3f}  {evr:>8.1f}%")
        n_pc = self.residuals.shape[0]
        n_bad = sum(self.statistics(m)["ks_p"] < 0.05 for m in range(n_pc))
        print(f"\n{n_bad} of {n_pc} components rejected by a KS test at the 5% "
              f"level (expected {0.05 * n_pc:.2f} of {n_pc} if the emulator were "
              "perfect).")
        print("A histogram narrower than N(0,1) means the predictive uncertainty "
              "is overestimated, a wider one that it is underestimated.")
        return self.residuals

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    @staticmethod
    def _hist_density(values, bins):
        """Histogram density normalized over the *whole* sample.

        ``ax.hist(density=True)`` normalizes over the points inside the bin
        range only, which would inflate the visible bars when a residual falls
        outside the plotted range; these constant weights make the bars
        integrate to one over the full sample instead.
        """
        weights = np.full(values.size, 1.0 / values.size)
        return np.histogram(values, bins=bins, weights=weights, density=True)[0]

    def plot(self, filepath=None):
        """Draw one histogram of z per principal component, overlaid with N(0,1)."""
        if self.residuals is None:
            raise ValueError("No residuals yet; run() first.")
        n_pc = self.residuals.shape[0]

        # shared binning and axis ranges, so the panels are directly comparable
        limit = max(XMIN_LIMIT,
                    min(ZPLOT_LIMIT, float(np.ceil(np.abs(self.residuals).max()))))
        bins = np.linspace(-limit, limit, N_BINS + 1)
        grid = np.linspace(-limit, limit, 400)
        peak = max(norm.pdf(grid).max(),
                   max(self._hist_density(self.residuals[m], bins).max()
                       for m in range(n_pc)))
        ymax = YMAX_HEADROOM * peak

        fig, axes = plt.subplots(1, n_pc,
                                 figsize=(PANEL_WIDTH_IN * n_pc, PANEL_HEIGHT_IN))
        # the caller owns this layout, so AutoPlot must not run its own
        # tight_layout / subplots_adjust pass (see AutoPlot.plot)
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.245, top=0.86,
                            wspace=0.08)

        for m, ax in enumerate(np.atleast_1d(axes)):
            z = self.residuals[m]
            s = self.statistics(m)
            evr = 100 * np.mean(self.explained_variance_ratio[:, m])
            n_out = int(np.sum(np.abs(z) > limit))
            caption = rf'$\mu={s["mean"]:.2f},\ \sigma={s["std"]:.2f}$'
            if n_out:
                caption += ('\n' + f'{n_out} point{"" if n_out == 1 else "s"} '
                            rf'outside $\pm{limit:g}$')
            ap = (AutoPlot()
                  .set_square(False)
                  .set_labels(rf'Normalized residual $z_{m + 1}$',
                              'Probability density' if m == 0 else '')
                  .set_title(f'PC {m + 1} ({evr:.1f}%)')
                  .set_limits((-limit, limit), (0.0, ymax))
                  .add_hist(z, bins=bins, density=True,
                            weights=np.full(z.size, 1.0 / z.size),
                            color=HIST_COLOR, edgecolor=HIST_EDGE,
                            linewidth=HIST_LINE_WIDTH)
                  .add_array(grid, norm.pdf(grid),
                             color=GAUSS_COLOR, lw=GAUSS_LINE_WIDTH)
                  # both annotations sit in the empty top strip created by
                  # YMAX_HEADROOM, so they can never collide with the bars
                  .text(-limit * 0.96, ymax * 0.97, caption,
                        ha='left', va='top', color=ANNOT_COLOR,
                        fontsize=FS_ANNOT)
                  # key of the black reference curve, matched to it by color
                  .text(limit * 0.96, ymax * 0.97, r'$\mathcal{N}(0,1)$',
                        ha='right', va='top', color=GAUSS_COLOR,
                        fontsize=FS_ANNOT))
            ap.plot(ax=ax, show=False)
            if m > 0:
                ax.tick_params(labelleft=False)   # identical y range in all panels

        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        base = filepath or os.path.join(self.config.plot_output_dir,
                                        'Emulator_Normalized_Residuals')
        base = os.path.splitext(base)[0]
        for suffix in ('.png', '.pdf'):
            fig.savefig(base + suffix,
                        dpi=600 if suffix == '.png' else 300,
                        bbox_inches='tight', pad_inches=0.02)
            print(f"Normalized-residual histograms saved to {base}{suffix}")
        plt.close(fig)
        return fig


def main():
    mode = (f"{N_FOLDS}-fold cross-validation" if N_FOLDS > 1
            else f"hold-out split, test fraction {TEST_FRACTION}")
    print("=" * 70)
    print("Universal Bayesian Analyzer -- emulator validation")
    print(f"  design table   = {os.path.join(config.input_dir, config.parameter_file)}")
    print(f"  theory table   = {os.path.join(config.input_dir, config.theoretical_data_file)}")
    print(f"  PCA components = {config.n_pca_components}")
    print(f"  validation     = {mode}")
    print("=" * 70)

    validator = EmulatorValidator(config, n_folds=N_FOLDS,
                                  test_fraction=TEST_FRACTION,
                                  random_seed=RANDOM_SEED)
    validator.run()
    validator.report()
    validator.plot()


if __name__ == "__main__":
    main()
