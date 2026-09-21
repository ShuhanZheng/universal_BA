# Class.py --- Universal Bayesian Analyzer core
# (DataLoader / Interpolator / MCMCSampler / OutputAnalyzer)
#
# The framework follows arXiv:2512.11536 (Phys. Rev. C 114, 024912 (2026)).
# Relative to the analysis code of that paper, it is generalized as follows:
#   1. All settings are driven by the `config` instance in Input.py (no hardcoding);
#   2. Interpolator gains an `analytical` option: when an analytical emulator
#      (with no theoretical error) is available, the PCA+GP emulator is skipped;
#   3. OutputAnalyzer is standardized: it keeps the per-chain samples, prints
#      parameter estimates using the parameter names from Input.py, draws a
#      Markov-chain quick check (one figure for every chain), and delegates the
#      joint-distribution plot to the external plot.py.

import os
import sys
import subprocess
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from joblib import dump, load
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')


class DataLoader:
    """Data loading: experimental data + (optional) parameter set and
    theoretical data used to train the GP emulator."""

    def __init__(self, config):
        self.config = config
        self.parameters = None
        self.theoretical_data = None
        self.experimental_data = None
        self.experimental_sigma_stat = None
        self.experimental_sigma_syst = None
        self.Cov_exp = None

    def _path(self, filename):
        return os.path.join(self.config.input_dir, filename)

    def load(self):
        """Load all data. Theoretical training data is skipped when analytical=True."""
        if not os.path.isdir(self.config.input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {self.config.input_dir}")

        self.experimental_data = np.loadtxt(self._path(self.config.experimental_data_file))
        self.experimental_sigma_stat = np.loadtxt(self._path(self.config.experimental_sigma_stat_file))
        self.experimental_sigma_syst = np.loadtxt(self._path(self.config.experimental_sigma_syst_file))
        self.Cov_exp = np.diag(self.experimental_sigma_stat) + self.experimental_sigma_syst

        print(f"experimental data  y_exp:       {self.experimental_data.shape}")
        print(f"statistical (diag variance):    {self.experimental_sigma_stat.shape}")
        print(f"systematic (covariance matrix): {self.experimental_sigma_syst.shape}")
        print(f"Sigma_exp = diag(sigma_stat^2) + Sigma_syst: {self.Cov_exp.shape}")

        if not self.config.analytical:
            self.parameters = np.loadtxt(self._path(self.config.parameter_file))
            self.theoretical_data = np.loadtxt(self._path(self.config.theoretical_data_file))
            print(f"parameter set {{x_n}}:         {self.parameters.shape}")
            print(f"theoretical data {{y_th(x_n)}}: {self.theoretical_data.shape}")
        else:
            print("analytical=True: skip loading of parameter set and theoretical data "
                  "(the analytical emulator directly returns y(x)).")
        return self


class Interpolator:
    """Provider of the observable mapping y(x) and the theoretical covariance
    Sigma_th(x).

    - analytical=True : call the analytical model provided in Input.py;
      theoretical error is zero, i.e. Sigma_th = 0;
    - analytical=False: PCA dimensionality reduction + a GP trained per
      principal component, yielding an emulator with theoretical uncertainty.
    """

    def __init__(self, config):
        self.config = config
        self.analytical = config.analytical
        self.n_pca_components = config.n_pca_components
        self.pca = None
        self.gpr_model = []
        # Cache of the x-independent pieces needed by predict_with_grad
        # (kernel matrix inverse, de-meaned training targets, ...). Built
        # lazily so that both the fit() and the load_model() paths are covered.
        self._grad_cache = None

    def fit(self, x_data, y_data):
        """Train the PCA+GP emulator on the parameter set {x_n} and the
        theoretical data {y_th(x_n)}."""
        if self.analytical:
            print("analytical=True: no need to train a PCA+GP emulator.")
            return

        self._grad_cache = None
        # The GP input dimension must equal what chi_square actually feeds it
        # (sampled parameters with the fixed ones appended).
        expected_inputs = self.config.N_parameter + len(self.config.fixed_params)
        if x_data.shape[1] != expected_inputs:
            raise ValueError(
                f"Parameter file has {x_data.shape[1]} columns but N_parameter "
                f"({self.config.N_parameter}) + len(fixed_params) "
                f"({len(self.config.fixed_params)}) = {expected_inputs}.")
        n_outputs = y_data.shape[1]
        if self.n_pca_components > n_outputs:
            print(f"n_pca_components ({self.n_pca_components}) exceeds the output "
                  f"dimension ({n_outputs}); reset to {n_outputs}")
            self.n_pca_components = n_outputs

        self.pca = PCA(n_components=self.n_pca_components)
        Y_pca = self.pca.fit_transform(y_data)
        print("PCA eigenvalues:", self.pca.explained_variance_)

        for dim in range(self.n_pca_components):
            kernel = (ConstantKernel(10, (1e-1, 1e6)) *
                      RBF(1.0, (1e-2, 1e6)) +
                      WhiteKernel(1e-3, (1e-5, 1e-1)))
            gp = GPR(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
            gp.fit(x_data, Y_pca[:, dim].reshape(-1, 1))
            self.gpr_model.append(gp)
            print(f"\nPC {dim + 1}: "
                  f"Constant={gp.kernel_.k1.k1.constant_value:.5f}, "
                  f"RBF length={gp.kernel_.k1.k2.length_scale:.15f}, "
                  f"White noise={gp.kernel_.k2.noise_level:.2e}")

        print("Model training finished.")

    def predict(self, x):
        """Return (mean y(x), theoretical covariance Sigma_th(x))."""
        if self.analytical:
            y = np.asarray(self.config.analytical_model(x), dtype=float).flatten()
            d = y.size
            return y, np.zeros((d, d))

        Y_pred_pca = []
        Y_std_pca = []
        for gp in self.gpr_model:
            mean, std = gp.predict([x], return_std=True)
            Y_pred_pca.append(mean[0])
            Y_std_pca.append(std[0])

        Y_std_pca = np.array(Y_std_pca)
        Unitary = self.pca.components_
        mu = self.pca.mean_

        Y_pred_original = np.array(Y_pred_pca) @ Unitary + mu

        return Y_pred_original.flatten(), self.sigma_th_from_var(Y_std_pca ** 2)

    def sigma_th_from_var(self, var_pca):
        """Decode a PC-space variance vector into Sigma_th (paper Eq. 448/450),

            Sigma_th = 2 * U^dagger diag(var_pca) U,   U^dagger = pca.components_.T.

        Paper convention: Sigma_th(x) = 2 * Sigma_Gauss(x), where the factor 2
        accounts for the uncertainty of the Schroedinger evolution framework
        (arXiv:2512.11536, Eq. for Sigma_th). Do not drop it anywhere, otherwise
        the parameter-dependent part of the covariance (and hence the exact
        variance-inflation term d*ln(f) and the analytic gradient) changes.
        """
        Unitary = self.pca.components_
        return 2.0 * (Unitary.T * var_pca) @ Unitary

    @staticmethod
    def _gp_normalization(gp):
        """Mean/std with which sklearn's normalize_y rescaled the training target.

        fit() uses normalize_y=True, so the GP is trained on (P - mean)/std and
        predict() undoes that map afterwards (sklearn _gpr.py, lines 447 and 488).
        The analytic derivatives must carry the same rescaling to stay equal to
        the derivative of the y and Sigma_th actually returned by predict().
        """
        y_mean = float(np.ravel(getattr(gp, "_y_train_mean", [0.0]))[0])
        y_std = float(np.ravel(getattr(gp, "_y_train_std", [1.0]))[0])
        return y_mean, y_std

    def _ensure_gradient_cache(self):
        """Precompute the x-independent quantities of the analytic gradient
        (derivation §3.3), once, for each principal component m:

          X_train  : training points x_p (Latin-hypercube design)
          K_inv    : K_m^{-1}, with K_m the training kernel including the white
                     noise (paper Eq. 459-463); the 1e-10 ridge that sklearn adds
                     internally is kept so that K_inv matches predict() exactly
          K_inv_P  : K_m^{-1} (P_q - mean), i.e. the inner sum over q in Eq. (2.4)
          l_m      : RBF length scale of Eq. (481)
          y_std    : normalize_y rescaling of the target (see _gp_normalization)
        """
        if self._grad_cache is not None:
            return self._grad_cache
        if self.analytical or not self.gpr_model or self.pca is None:
            raise ValueError("The PCA+GP emulator has not been trained yet; "
                             "cannot build the analytic gradient.")
        cache = []
        for gp in self.gpr_model:
            X_train = np.asarray(gp.X_train_, dtype=float)
            K = gp.kernel_(X_train)
            K_inv = np.linalg.inv(K + gp.alpha * np.eye(K.shape[0]))
            y_mean, y_std = self._gp_normalization(gp)
            # undo the normalize_y mapping to recover the raw PC targets P_q
            P_train = np.asarray(gp.y_train_, dtype=float).ravel() * y_std + y_mean
            try:
                length_scale = float(gp.kernel_.k1.k2.length_scale)
            except AttributeError as exc:
                raise AttributeError(
                    "predict_with_grad assumes the kernel built in fit(): "
                    "ConstantKernel * RBF + WhiteKernel.") from exc
            cache.append({
                "X_train": X_train,
                "K_inv": K_inv,
                "K_inv_P": K_inv @ (P_train - y_mean),  # P_q of Eq. (2.4) is de-meaned
                "length_scale": length_scale,
                "y_std": y_std,
            })
        print("Analytic-gradient cache built "
              f"({len(cache)} principal components, "
              f"{cache[0]['X_train'].shape[0]} training points).")
        self._grad_cache = cache
        return cache

    def predict_with_grad(self, x):
        """Same y and Sigma_th as predict(), plus their analytic derivatives.

        Returns (y, Sigma_th, dy/dx, dsigma2/dx) with, for a full input vector x
        (sampled parameters with the fixed parameters appended):

          y          (N,)                : mean prediction, Eq. (437)/(2.3)
          Sigma_th   (N, N)              : 2 U^dagger diag(sigma2) U, Eq. (2.6)
          dy/dx      (N, d)              : U^dagger dP/dx_i, Eq. (2.3)
          dsigma2/dx (n_pc, d)           : d sigma^2_{P,m}/dx_i, Eq. (2.5)

        Derivatives are taken with respect to every component of x, including the
        fixed ones; the caller selects the sampled components. Values come from
        gp.predict, so they are identical to predict()'s; only the derivatives
        are new. PCA+GP emulator only (analytical=False).
        """
        if self.analytical:
            raise RuntimeError("predict_with_grad is implemented for the PCA+GP "
                               "emulator only (analytical=False).")
        cache = self._ensure_gradient_cache()
        x = np.asarray(x, dtype=float)

        Y_pred_pca = []
        Y_std_pca = []
        dP_dx_pca = []
        dvar_dx_pca = []
        for gp, cached in zip(self.gpr_model, cache):
            mean, std = gp.predict([x], return_std=True)
            Y_pred_pca.append(mean[0])
            Y_std_pca.append(std[0])

            X_train = cached["X_train"]
            l2 = cached["length_scale"] ** 2
            kappa = np.asarray(gp.kernel_(x[None, :], X_train), dtype=float)[0]

            # Eq. (2.4): dPbar_m/dx_i = -(1/l^2) sum_p (x-x_p)_i kappa_p (K^-1 P)_p
            v = kappa * cached["K_inv_P"]
            dP_dx_pca.append(-(x * v.sum() - v @ X_train) / l2)

            # Eq. (2.5): d sigma^2/dx_i = (2/l^2) sum_p (x-x_p)_i kappa_p (K^-1 kappa)_p
            # The y_std^2 factor is the unit conversion of _gp_normalization: it
            # turns the kernel-space variance into the physical-units variance
            # that gp.predict(return_std=True) -- and hence chi_square -- uses.
            g = kappa * (cached["K_inv"] @ kappa)
            dvar_dx_pca.append(2.0 * cached["y_std"] ** 2 * (x * g.sum() - g @ X_train) / l2)

        Y_pred_pca = np.array(Y_pred_pca)
        dP_dx_pca = np.array(dP_dx_pca)                 # (n_pc, d)

        Unitary = self.pca.components_
        y = Y_pred_pca @ Unitary + self.pca.mean_
        Cov_th = self.sigma_th_from_var(np.array(Y_std_pca) ** 2)
        dy_dx = Unitary.T @ dP_dx_pca                   # (N, d)
        return y.flatten(), Cov_th, dy_dx, np.array(dvar_dx_pca)

    def save_model(self, filepath):
        if self.analytical:
            print("analytical=True: no model to save.")
            return
        if self.gpr_model is None or self.pca is None:
            raise ValueError("The model has not been trained yet; cannot save.")
        dump((self.gpr_model, self.pca), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if self.analytical:
            print("analytical=True: no model to load.")
            return
        self.gpr_model, self.pca = load(filepath)
        self._grad_cache = None
        print(f"Model loaded from {filepath}.")


class MCMCSampler:
    """Hamiltonian Monte Carlo (HMC) sampler. The sampling logic follows the
    scheme of arXiv:2512.11536."""

    def __init__(self, interpolator, data_loader, config):
        self.interpolator = interpolator
        self.data_loader = data_loader
        self.config = config
        self.bounds = config.Bayesian_bound
        self.fixed_params = np.asarray(config.fixed_params, dtype=float)
        self.burn_in = config.burn_in
        self.decorrelation_length = config.decorrelation_length
        self.Cov_exp = data_loader.Cov_exp
        self.samples = []
        self.Nx_values = []
        self.epsilon = np.full(len(self.bounds), config.gradient_epsilon)
        # analytical=True : explicit emulator, no PCA+GP bookkeeping -> the
        # potential gradient stays a cheap scipy finite difference.
        # analytical=False: PCA+GP emulator -> analytic gradient (see gradient).
        self.analytical = bool(getattr(config, "analytical", False))
        # Variance inflation: the N_data observables are treated as
        # non-independent measurements, so -2lnL is scaled by 1/f
        # (see chi_square). Dividing the potential by f also divides the
        # curvature by f, which relaxes the leapfrog stability limit on dt.
        self.variance_inflation = float(getattr(config, 'variance_inflation', 1.0))
        self._f_final = self.variance_inflation
        # Simulated-annealing burn-in factor (see sample); 1.0 disables it.
        self.annealing_factor = float(getattr(config, 'annealing_factor', 1.0))
        # In analytical mode the emulator has no theoretical uncertainty, so
        # Cov = Cov_exp is parameter-independent: precompute its inverse and
        # log-determinant once (mathematically identical, much faster).
        self._const_cov = self.analytical
        if self._const_cov:
            self._cov_inv = np.linalg.inv(self.Cov_exp)
            sign, self._logdet = np.linalg.slogdet(self.Cov_exp)
            if sign <= 0:
                raise ValueError("Cov_exp is not positive definite.")

    def wrap_parameters(self, params):
        """Periodic boundary condition that keeps parameters inside Bayesian_bound."""
        wrapped = np.copy(params)
        for i, (low, high) in enumerate(self.bounds):
            if wrapped[i] < low:
                wrapped[i] = high - ((low - wrapped[i]) % (high - low))
            elif wrapped[i] > high:
                wrapped[i] = low + ((wrapped[i] - high) % (high - low))
        return wrapped

    def chi_square(self, x):
        """Negative log-likelihood under a flat prior, with variance inflation
        by the factor f = config.variance_inflation (Sigma -> f*Sigma):
        chi^2 = [Delta y^T Sigma^{-1} Delta y]/f + ln det(Sigma) + d*ln(f).

        The potential is evaluated at exactly the point it is given: there is NO
        periodic wrapping here. Boundary handling lives entirely in `sample`,
        whose hard-wall reflection keeps the trajectory inside Bayesian_bound,
        so every point reaching this function is already in the prior box.
        Wrapping internally would instead corrupt the numerical gradient:
        approx_fprime probes x + epsilon*e_i, and when x lies within epsilon of
        a face that probe crosses onto the OPPOSITE face, turning a correct
        one-sided difference quotient into a cliff of order
        |chi^2(near face) - chi^2(far face)|/epsilon (order 1e10 in the current
        setup). The kick it delivers ejects the trajectory far outside the box
        (millions of wall reflections, costing ~0.5 s) and the proposal is then
        always rejected. Returning inf or raising for out-of-range input would
        fail for the same reason: that face-crossing probe is legitimate."""
        x = np.array(x, dtype=float)
        x_full = np.append(x, self.fixed_params)
        mu_x, Cov_x = self.interpolator.predict(x_full)
        delta_y = mu_x - self.data_loader.experimental_data
        f = self.variance_inflation
        if self._const_cov:
            # Constant covariance: ln det(Sigma) + d*ln(f) is parameter-
            # independent, so dividing the whole expression by f is proper
            # variance inflation up to an irrelevant constant (only chi^2
            # differences enter the HMC acceptance test).
            return (delta_y @ self._cov_inv @ delta_y.T + self._logdet) / f
        Cov = Cov_x + self.Cov_exp
        try:
            Cov_inv = np.linalg.inv(Cov)
        except np.linalg.LinAlgError:
            print('Cannot invert the covariance matrix; return a large chi^2.')
            return np.inf
        chi_1 = delta_y @ Cov_inv @ delta_y.T
        try:
            chi_2 = np.log(np.linalg.det(Cov))
        except np.linalg.LinAlgError:
            return np.inf
        # Parameter-dependent covariance: use the exact inflation formula.
        # Simply dividing (chi_1 + chi_2) by f would wrongly flatten the
        # x-dependence of the ln det term and distort the posterior.
        d = Cov.shape[0]
        return chi_1 / f + chi_2 + d * np.log(f)

    def _potential_gradient(self, x):
        """Gradient of the potential V(x) = 0.5*chi_square(x) at x.

        analytical=True : the emulator is an explicit function and there is no
            PCA+GP bookkeeping, so the potential stays cheap and the scipy
            difference quotient is kept.
        analytical=False: analytic gradient from the PCA+GP emulator (gradient),
            which needs one emulator evaluation instead of d+1.
        """
        if self.analytical:
            return approx_fprime(xk=x, f=lambda z: 0.5 * self.chi_square(z),
                                 epsilon=self.epsilon)
        return self.gradient(x)

    def gradient(self, x):
        """dV/dx_i for the PCA+GP emulator (derivation §2.7 / §3.1):

            dV/dx_i = (1/f)[ dy^T S^-1 dy/dx_i - (1/2) dy^T S^-1 dS/dx_i S^-1 dy ]
                      + (1/2) tr(S^-1 dS/dx_i),

        with S = Sigma_th(x) + Sigma_exp and, from Eq. (2.6),

            dS/dx_i = dSigma_th/dx_i = 2 U^dagger diag(dsigma2_{P,m}/dx_i) U.

        The 1/f factor multiplies the residual terms only, exactly as in
        chi_square; the trace term is not divided by f. x is the sampled
        parameter vector; the fixed parameters are appended as in chi_square.
        """
        if self.analytical:
            raise RuntimeError("The analytic gradient is implemented for the "
                               "PCA+GP emulator only (analytical=False); "
                               "use approx_fprime for analytical=True.")
        x = np.array(x, dtype=float)
        x_full = np.append(x, self.fixed_params)
        mu_x, Cov_x, dy_dx, dvar_dx = self.interpolator.predict_with_grad(x_full)
        delta_y = mu_x - self.data_loader.experimental_data
        Cov = Cov_x + self.Cov_exp
        try:
            Cov_inv = np.linalg.inv(Cov)
        except np.linalg.LinAlgError:
            # Same convention as chi_square: an invertible covariance is not
            # guaranteed far outside the posterior mode. The potential is +inf
            # there, so the proposal is rejected either way; a zero gradient
            # keeps the chain alive instead of aborting the whole run.
            print('Cannot invert the covariance matrix; return a zero gradient.')
            return np.zeros(x.size)
        f = self.variance_inflation

        grad = np.empty(x.size)
        for i in range(x.size):
            dCov = self.interpolator.sigma_th_from_var(dvar_dx[:, i])  # Eq. (2.6)
            residual = delta_y @ Cov_inv @ dy_dx[:, i]
            quadratic = delta_y @ Cov_inv @ dCov @ Cov_inv @ delta_y
            trace = np.trace(Cov_inv @ dCov)
            grad[i] = (residual - 0.5 * quadratic) / f + 0.5 * trace
        return grad

    def sample(self, x0, delta_t, T, iterations):
        """Sample a single HMC chain.

        Simulated-annealing burn-in: while i <= burn_in the variance inflation
        is temporarily raised to f_final * annealing_factor and geometrically
        relaxed back to f_final. This flattens the high-chi^2 'desert' so that
        chains started anywhere in the prior box can descend to the posterior
        mode. (Raw HMC cannot perform this transition: the energy drop of
        O(100) between desert and mode is converted into kinetic energy, the
        fixed-dt integrator can no longer resolve the trajectory, and every
        proposal fails the Metropolis test -- acceptance is exactly 0.)
        After burn-in the inflation equals f_final exactly, so the sampled
        target distribution is untouched. annealing_factor=1 disables this."""
        current_x = np.copy(x0)
        Nx_present = self.chi_square(current_x)
        self.samples = []
        self.Nx_values = []
        self.n_accept = 0

        for i in range(iterations + 1):
            if (self.annealing_factor != 1.0 and self.burn_in > 0
                    and i <= self.burn_in):
                self.variance_inflation = self._f_final * self.annealing_factor ** (
                    1.0 - i / self.burn_in)
                Nx_present = self.chi_square(current_x)
            # draw a fresh momentum
            p_t = np.random.normal(loc=0, scale=1, size=current_x.shape)
            # standard HMC: the FRESH momentum of this iteration enters
            # H_present in the Metropolis test (a stale momentum from a
            # previous trajectory inflates the acceptance probability and
            # can fling the chain into stiff regions where it freezes)
            p_initial = np.copy(p_t)
            x_t = np.copy(current_x)

            # Leapfrog integration of Hamilton's canonical equations; the
            # potential gradient is analytic for the PCA+GP emulator and a
            # scipy difference quotient for the analytic emulator (see
            # _potential_gradient). Both are evaluated at exactly the point
            # handed to them -- the boundary reflection below mirrors the
            # position, so the gradient is always taken inside the prior box.
            for _ in range(int(T)):
                grad_xt = self._potential_gradient(x_t)
                p_t_plus_halfdelta = p_t - (delta_t / 2) * grad_xt
                x_t_plus_delta = x_t + delta_t * p_t_plus_halfdelta
                # Hard-wall reflection at the prior-box faces: the flat prior
                # truncates the posterior at the bounds, so a trajectory that
                # crosses a face is bounced back (position mirrored, normal
                # momentum reversed). This is reversible, volume-preserving and
                # exactly energy-conserving, unlike the periodic wrap, which
                # creates chi^2 cliffs at the faces (chi^2 at x=low generally
                # differs from chi^2 at x=high): every wrap crossing injects an
                # O(cliff) error into the Hamiltonian and the proposal is
                # rejected -- chains starting within one trajectory-length of a
                # face then freeze with acceptance exactly 0.
                for j, (low, high) in enumerate(self.bounds):
                    while x_t_plus_delta[j] < low or x_t_plus_delta[j] > high:
                        if x_t_plus_delta[j] < low:
                            x_t_plus_delta[j] = 2 * low - x_t_plus_delta[j]
                        else:
                            x_t_plus_delta[j] = 2 * high - x_t_plus_delta[j]
                        p_t_plus_halfdelta[j] = -p_t_plus_halfdelta[j]
                grad_xt_plus_delta = self._potential_gradient(x_t_plus_delta)
                p_t_plus_delta = p_t_plus_halfdelta - (delta_t / 2) * grad_xt_plus_delta
                x_t = x_t_plus_delta
                p_t = p_t_plus_delta

            # wrap the parameters back into the bounds
            x_new = self.wrap_parameters(x_t)
            Nx_new = self.chi_square(x_new)

            # acceptance probability
            Hamiltonian_present = 0.5 * (Nx_present + p_initial @ p_initial.T)
            Hamiltonian_new = 0.5 * (Nx_new + p_t @ p_t.T)
            accept_probability = min(1, np.exp(-(Hamiltonian_new - Hamiltonian_present)))
            if not np.isfinite(accept_probability):
                # reject non-finite proposals so a bad trajectory can never
                # contaminate the current state of the chain
                accept_probability = 0.0

            if np.random.rand() < accept_probability:
                current_x = x_new
                Nx_present = Nx_new
                self.n_accept += 1

            if i > self.burn_in and (i - self.burn_in) % self.decorrelation_length == 0:
                self.samples.append(current_x)
                self.Nx_values.append(Nx_present)

    def get_samples(self):
        """Return (samples, chi^2 values) of the chain."""
        return np.array(self.samples), np.array(self.Nx_values)


class OutputAnalyzer(MCMCSampler):
    """Analysis and plotting of the sampling results: parallel chains,
    per-chain save/load, Markov-chain quick check, and delegation of the
    joint-distribution plot to the external plot.py."""

    def __init__(self, interpolator, data_loader, config):
        super().__init__(interpolator, data_loader, config)
        self.all_samples = []
        self.all_Nx_values = []
        self.chain_samples = []      # samples of each chain (list of 2D arrays)
        self.chain_Nx_values = []    # chi^2 values of each chain (list of 1D arrays)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def run_single_chain(self, x0, delta_t, T, iterations):
        sampler = MCMCSampler(self.interpolator, self.data_loader, self.config)
        sampler.sample(x0=x0, delta_t=delta_t, T=T, iterations=iterations)
        samples, Nx = sampler.get_samples()
        accept_rate = sampler.n_accept / max(iterations, 1)
        return samples, Nx, accept_rate

    def run_parallel_sampling(self, initial_x0, delta_t, T, iterations, n_chains):
        """Run n_chains independent HMC chains in parallel and collect all samples."""
        if len(initial_x0) != n_chains:
            raise ValueError(f"len(initial_x0) ({len(initial_x0)}) does not match "
                             f"n_chains ({n_chains}).")
        print(f"Start parallel sampling ({n_chains} chains)...")
        results = Parallel(n_jobs=n_chains)(
            delayed(self.run_single_chain)(x0, delta_t, T, iterations) for x0 in initial_x0
        )
        self.chain_samples = []
        self.chain_Nx_values = []
        for chain_idx, (samples, Nx, accept_rate) in enumerate(results):
            self.chain_samples.append(samples)
            self.chain_Nx_values.append(Nx)
            print(f"Chain {chain_idx + 1}: {len(samples)} samples, "
                  f"acceptance rate = {accept_rate:.3f}.")
        self.all_samples = np.vstack(self.chain_samples)
        self.all_Nx_values = np.hstack(self.chain_Nx_values)
        print("All chains have been merged.")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_all_samples(self, filepath=None):
        """Save the merged samples (columns: x_1..x_d, chi^2)."""
        filepath = filepath or self.config.sample_filepath
        if not self.all_samples.size:
            raise ValueError("No samples to save. Run sampling first.")
        data = np.hstack((self.all_samples, self.all_Nx_values.reshape(-1, 1)))
        np.savetxt(filepath, data, fmt='%.6f', delimiter=' ')
        print(f"Merged samples saved to {filepath}")

    def load_all_samples(self, filepath=None):
        """Load the merged samples; the last column is the chi^2 value."""
        filepath = filepath or self.config.sample_filepath
        data = np.loadtxt(filepath)
        self.all_samples = data[:, :-1]
        self.all_Nx_values = data[:, -1]
        print(f"Merged samples loaded from {filepath}.")
        return self.all_samples, self.all_Nx_values

    def save_all_chains(self, filepath=None):
        """Save the per-chain samples (columns: chain_id, x_1..x_d, chi^2)."""
        filepath = filepath or self.config.chain_filepath
        if not self.chain_samples:
            raise ValueError("No per-chain samples to save. Run parallel sampling first.")
        blocks = []
        for idx, samples in enumerate(self.chain_samples):
            Nx = self.chain_Nx_values[idx].reshape(-1, 1)
            chain_id = np.full((len(samples), 1), idx)
            blocks.append(np.hstack((chain_id, samples, Nx)))
        np.savetxt(filepath, np.vstack(blocks), fmt='%.6f', delimiter=' ')
        print(f"Per-chain samples saved to {filepath}")

    def load_all_chains(self, filepath=None):
        """Load the per-chain samples. Falls back to treating the merged
        samples as a single chain if the per-chain file is missing."""
        filepath = filepath or self.config.chain_filepath
        if not os.path.exists(filepath):
            print(f"Per-chain file {filepath} not found; falling back to treating "
                  "the merged samples as a single chain.")
            if self.all_samples.size:
                self.chain_samples = [self.all_samples]
                self.chain_Nx_values = [self.all_Nx_values]
            return
        data = np.loadtxt(filepath)
        n = self.config.N_parameter
        ids = data[:, 0].astype(int)
        self.chain_samples = []
        self.chain_Nx_values = []
        for idx in np.unique(ids):
            mask = ids == idx
            self.chain_samples.append(data[mask, 1:1 + n])
            self.chain_Nx_values.append(data[mask, 1 + n])
        print(f"Per-chain samples loaded from {filepath} "
              f"({len(self.chain_samples)} chains).")

    # ------------------------------------------------------------------
    # Statistics (driven by the parameter names in Input.py)
    # ------------------------------------------------------------------
    def calculate_medians(self):
        medians = np.median(self.all_samples, axis=0)
        for name, m in zip(self.config.parameter_names, medians):
            print(f"Median of {name}: {m:.4f}")
        return medians

    def calculate_95ci(self):
        lower = np.percentile(self.all_samples, 2.5, axis=0)
        upper = np.percentile(self.all_samples, 97.5, axis=0)
        for name, lo, hi in zip(self.config.parameter_names, lower, upper):
            print(f"95% CI of {name}: [{lo:.4f}, {hi:.4f}]")
        return lower, upper

    def calculate_68ci(self):
        lower = np.percentile(self.all_samples, 16, axis=0)
        upper = np.percentile(self.all_samples, 84, axis=0)
        for name, lo, hi in zip(self.config.parameter_names, lower, upper):
            print(f"68% CI of {name}: [{lo:.4f}, {hi:.4f}]")
        return lower, upper

    def report_summary(self):
        """Print a summary table of median values and 68%/95% credible intervals."""
        medians = self.calculate_medians()
        lo68, hi68 = self.calculate_68ci()
        lo95, hi95 = self.calculate_95ci()
        print("\n========== Parameter estimation summary ==========")
        for name, m, a, b, c, e in zip(self.config.parameter_names, medians,
                                       lo68, hi68, lo95, hi95):
            print(f"{name:>10}  median={m:.4f}  68% CI=[{a:.4f},{b:.4f}]  "
                  f"95% CI=[{c:.4f},{e:.4f}]")
        return medians

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_markov_chains(self, dimension=0, filepath=None):
        """Draw one dimension (by default the first parameter) of every chain
        in a single figure, for a quick convergence check.

        Styled with AutoSciPlot (Times New Roman / inward ticks, same as the
        other figures). AutoSciPlot.py + SciPlotStyle.py ship inside
        universal_BA/ so the package stays self-contained."""
        if not self.chain_samples:
            raise ValueError("No per-chain samples to plot. Run parallel sampling "
                             "or load the per-chain file first.")
        os.makedirs(self.config.plot_output_dir, exist_ok=True)
        filepath = filepath or os.path.join(self.config.plot_output_dir,
                                            'Markov_Chains_Check.png')

        # local import: SciPlotStyle sets global rcParams when imported, so
        # it must not leak into this module's import-time side effects
        from AutoSciPlot import AutoPlot

        ap = (AutoPlot()
              .set_figsize(12, 6)
              .set_square(False)  # wide trace plot, not a square panel
              .set_labels('Sample index (decorrelated)',
                          self.config.parameter_names[dimension])
              .set_legend(loc='best', frameon=False,
                          ncol=max(1, len(self.chain_samples) // 2)))
        for idx, chain in enumerate(self.chain_samples):
            ap.add_array(np.arange(len(chain)), chain[:, dimension],
                         label=f'Chain {idx + 1}', lw=1.5, alpha=0.85)
        fig, _ = ap.plot(save_path=filepath, show=False)
        plt.close(fig)
        print(f"Markov-chain check figure saved to {filepath}")

    def plot_joint_distributions(self):
        """Delegate the joint-distribution plot to the external plot.py,
        so that it can also be run standalone and its style easily edited."""
        if not os.path.exists(self.config.sample_filepath):
            raise ValueError(f"Sample file {self.config.sample_filepath} not found; "
                             "run sampling or load samples first.")
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot.py')
        print("Executing external plotting script plot.py ...")
        subprocess.run([sys.executable, script], check=True)
