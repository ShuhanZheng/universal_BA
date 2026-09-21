# Input.py --- Universal Bayesian Analysis: input configuration
#
# This is the only file the user needs to edit for a given application.
# All other modules (Class.py / Analysis.py / plot.py) read the shared
# `config` instance via `from Input import config`.
#
# All data-file paths are relative, resolved against the working directory
# from which Analysis.py / plot.py are launched (i.e. universal_BA/).
#
# Main inputs:
#   * y_exp and Sigma_exp (statistical + systematic) -- read by DataLoader
#     from the file names below;
#   * N_parameter   -- total number of parameters to be estimated (d);
#   * Bayesian_bound -- range of each parameter; its length MUST equal
#     N_parameter (a mismatch raises an error);
#   * parameter_names -- name of each parameter (LaTeX format); its length
#     MUST equal N_parameter;
#   * analytical    -- whether to use an analytical emulator (no theoretical
#     error). If True, analytical_model must be provided.
# The remaining HMC-sampling / emulator / output-path settings carry the
# defaults used by the shipped example; see the field comments below.

from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# Placeholder for the analytical emulator
# ============================================================================
def _default_analytical_model(x):
    """Default analytical emulator; must be overridden in Input.py when analytical=True."""
    raise NotImplementedError(
        "analytical=True requires an analytical emulator.\n"
        "Define analytical_model(x) in Input.py and pass it to config, e.g.\n"
        "    config = BayesianInput(analytical=True, analytical_model=my_model, ...)\n"
        "Here x is the full parameter vector (with the fixed parameters appended "
        "at the end) and the function returns the model output vector y(x)."
    )


@dataclass
class BayesianInput:
    # ============================ Data files (DataLoader) ============================
    input_dir: str = './input'
    parameter_file: str = 'bottomonium_param.dat'                    # parameter set {x_n}: N_parameter + len(fixed_params) columns
    theoretical_data_file: str = 'filtered_bottomonium_results.dat'  # theoretical observables {y_th(x_n)}: one observable vector per row
    experimental_data_file: str = 'filtered_exp_data.dat'            # y_exp: 1D array
    experimental_sigma_stat_file: str = 'filtered_exp_stat.dat'      # Sigma_exp^stat: diagonal variances (1D array)
    experimental_sigma_syst_file: str = 'syst_cov_matrix.dat'        # Sigma_exp^syst: non-diagonal covariance matrix (2D array)

    # ============================ Bayesian parameter definitions ============================
    N_parameter: int = 5                      # number of parameters to be estimated (d)
    Bayesian_bound: list = field(default_factory=lambda: [
        (0.0, 3.0), (0.5, 3.0), (0.0, 1.0), (0.0, 1.0), (1.5, 3.0),
    ])  # range of each parameter; length MUST equal N_parameter
    parameter_names: list = field(default_factory=lambda: [
        r'$a_m$', r'$f_T$', r'$f_1$', r'$f_2$', r'$f_3$',
    ])  # name of each parameter (LaTeX format); length MUST equal N_parameter
    fixed_params: list = field(default_factory=lambda: [0.18])      # parameters appended at the end of x (e.g. switching temperature T_d), kept fixed during sampling

    # ============================ Emulator options ============================
    analytical: bool = False                   # True: use analytical emulator (no theoretical error), skip PCA+GP; False: train a PCA+GP emulator with uncertainties
    analytical_model: callable = field(default=_default_analytical_model, repr=False)  # analytical model y=f(x); x includes the fixed parameters at the end
    n_pca_components: int = 5                 # number of PCA components retained (only used when analytical=False)

    # ============================ HMC sampling settings ============================
    delta_t: float = 0.02                     # Leapfrog time step. Measured curvature at f=125:
                                              # omega_max ~ 22 at the mode, ~95 in the low-f_n desert
                                              # (stability limits dt < 0.09 / 0.02). The annealed
                                              # burn-in keeps chains away from the stiff desert, so
                                              # dt only needs to resolve the near-mode dynamics
    leapfrog_steps: int = 100                 # number of Leapfrog steps per trajectory (T)
    target_number: int = 10                   # target number of samples per chain
    burn_in: int = 1000                       # burn-in steps (discarded)
    decorrelation_length: int = 100           # decorrelation length (keep one sample every such steps)
    n_chains: int = 8                         # number of parallel Markov chains
    random_seed: int = 42                     # random seed (reproducibility)
    gradient_epsilon: float = 1e-8            # step used by scipy approx_fprime for numerical gradients
    annealing_factor: float = 1.0             # simulated-annealing burn-in: the variance inflation
                                              # starts at variance_inflation*annealing_factor and is
                                              # geometrically relaxed to variance_inflation over the
                                              # burn-in phase, letting chains started anywhere in the
                                              # prior box descend to the posterior mode; 1.0 disables it

    # ============================ Likelihood settings ============================
    variance_inflation: float = 125.0         # conservative inflation of the data covariance: the
                                              # N_data observables are treated as non-independent
                                              # measurements, so -2lnL is scaled by 1/f:
                                              # chi2/f + ln det(Sigma) + d*ln(f)

    # ============================ Output file paths ============================
    model_filepath: str = 'gpr_pca_emulator.joblib'   # save/load path of the PCA+GP model
    sample_filepath: str = 'HMC_sample.dat'           # merged samples [x_1..x_d, chi^2]
    chain_filepath: str = 'HMC_sample_chains.dat'     # per-chain samples [chain_id, x_1..x_d, chi^2]
    plot_output_dir: str = './HMC_plot'               # output directory for figures
    figure_annotation: str = None                     # annotation text on the figure (e.g. r'$T_d = 160$ MeV'); None disables it

    # ============================ Derived quantities ============================
    @property
    def iterations(self) -> int:
        """Total number of sampling steps = target_number * decorrelation_length + burn_in."""
        return self.target_number * self.decorrelation_length + self.burn_in

    def validate(self) -> bool:
        """Validate the consistency of the input configuration."""
        if len(self.Bayesian_bound) != self.N_parameter:
            raise ValueError(
                f"Length of Bayesian_bound ({len(self.Bayesian_bound)}) "
                f"does not match N_parameter ({self.N_parameter})!"
            )
        if len(self.parameter_names) != self.N_parameter:
            raise ValueError(
                f"Number of parameter_names ({len(self.parameter_names)}) "
                f"does not match N_parameter ({self.N_parameter})!"
            )
        if self.analytical and self.analytical_model is _default_analytical_model:
            raise ValueError(
                "analytical=True but no analytical emulator provided: define "
                "analytical_model(x) in Input.py and pass it to config."
            )
        return True


# ============================================================================
# Active configuration: bottomonium ImV/T data from the server (input/Td018/)
# ============================================================================
# Emulator: PCA + one GP per principal component (analytical=False). This is
# the branch that carries the analytic HMC potential gradient derived in
# references/partialV_analytic_derivation.md (Class.py: Interpolator.predict_with_grad
# and MCMCSampler.gradient). There the theoretical covariance Sigma_th(x) does
# not vanish, so ln det Sigma(x) and its x-derivative contribute as well.
#
# Files in input/Td018/ (same layout as the server version):
#   bottomonium_param.dat             {x_n}: 200 design points, 5 sampled
#                                     parameters + 1 fixed parameter (last column)
#   filtered_bottomonium_results.dat  {y_th(x_n)}: 200 x 57 observables
#   filtered_exp_data.dat             y_exp: 57 values
#   filtered_exp_stat.dat             Sigma_exp^stat: 57 diagonal variances
#   syst_cov_matrix.dat               Sigma_exp^syst: 57 x 57 covariance
config = BayesianInput(
    # ---- data files ----
    input_dir='./input/Td018',
    parameter_file='bottomonium_param.dat',
    theoretical_data_file='filtered_bottomonium_results.dat',
    experimental_data_file='filtered_exp_data.dat',
    experimental_sigma_stat_file='filtered_exp_stat.dat',
    experimental_sigma_syst_file='syst_cov_matrix.dat',

    # ---- parameters: 5 sampled (names/bounds as in the server run) ----
    N_parameter=5,
    Bayesian_bound=[(0.0, 3.0), (0.5, 3.0), (0.0, 1.0), (0.0, 1.0), (1.5, 3.0)],
    parameter_names=[r'$a_m$', r'$f_T$', r'$f_1$', r'$f_2$', r'$f_3$'],
    fixed_params=[0.18],       # last column of bottomonium_param.dat (T_d)

    # ---- PCA+GP emulator (uses the analytic potential gradient) ----
    analytical=False,
    n_pca_components=5,

    # ---- output paths: kept separate from the analytic-emulator run so that
    #      neither the saved model nor the archived samples are overwritten ----
    model_filepath='gpr_pca_emulator_Td018.joblib',
    sample_filepath='HMC_sample_Td018.dat',
    chain_filepath='HMC_sample_chains_Td018.dat',
    plot_output_dir='./HMC_plot_Td018',
    figure_annotation=r'$T_d = 180$ MeV',

    # NOTE: the HMC settings (delta_t, leapfrog_steps, variance_inflation,
    # annealing_factor, ...) still carry the values tuned for the analytic
    # rT-lattice run and have NOT been retuned for this data set.
)


# ============================================================================
# Alternative configuration: analytic emulator on the rT lattice toy problem
# ============================================================================
# analytical=True skips PCA+GP entirely: Sigma_th = 0, the covariance is just
# Sigma_exp and is x-independent, and the potential gradient stays the scipy
# difference quotient (Class.py: MCMCSampler._potential_gradient). Kept as a
# template -- to run it, swap it in for the `config = ...` above.
#
# Model (dimensionless):  ImV/T = (rT)^{f_n} + f_l * (rT)
#   x = (f_n, f_l)  -- 2-dimensional parameter vector, no fixed parameters
#   y(x)            -- N_lattice = 125 model predictions, one per lattice rT
# The lattice rT values (all temperatures merged) are loaded once from
# input/rT_lattice.dat.
_R_T_LATTICE = np.loadtxt('./input/rT_lattice.dat')


def analytical_model(x):
    """Map x = (f_n, f_l) to the model vector y(x) = (rT)^{f_n} + f_l*(rT)."""
    f_n, f_l = x
    return _R_T_LATTICE**f_n + f_l * _R_T_LATTICE


# config = BayesianInput(
#     # ---- data files (input/) ----
#     experimental_data_file='exp_ImV.dat',             # y_exp: 125 values of ImV/T
#     experimental_sigma_stat_file='exp_ImV_stat.dat',  # diagonal variances unc_up^2
#     experimental_sigma_syst_file='exp_ImV_syst.dat',  # 125x125 zero matrix (no syst)

#     # ---- 2-parameter Bayesian estimation ----
#     N_parameter=2,
#     Bayesian_bound=[(0.0, 3.0), (0.0, 3.0)],
#     parameter_names=[r'$f_n$', r'$f_l$'],
#     fixed_params=[],

#     # ---- analytical emulator (no PCA+GP, no theoretical error) ----
#     analytical=True,
#     analytical_model=analytical_model,

#     # ---- HMC settings: 50 kept samples/chain x 8 chains = 400 samples ----
#     target_number=50,
#     # chains ALWAYS start from uniformly random points over the full prior box
#     # (Bayesian_bound): there is no separate warm-start box. Convergence from
#     # anywhere in the box is achieved by the annealed burn-in below: raw HMC
#     # cannot descend from the high-chi^2 desert (the ~250-unit energy drop
#     # makes every desert trajectory fail the Metropolis test -> the chain
#     # freezes with acceptance exactly 0), whereas with the inflation ramp
#     # all four corner starts reach the mode with acceptance ~0.9
#     annealing_factor=100.0,
# )


if __name__ == "__main__":
    config.validate()
    print("=== Universal Bayesian Analysis: input configuration ===")
    print(f"  N_parameter  = {config.N_parameter}")
    print(f"  parameters   = {config.parameter_names}")
    print(f"  bounds       = {config.Bayesian_bound}")
    print(f"  fixed_params = {config.fixed_params}")
    print(f"  emulator     = {'analytical' if config.analytical else 'PCA+GP'}")
    print(f"  iterations   = {config.iterations}")
    print("  configuration validated.")
