# universal_BA — general-purpose Bayesian parameter estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

A self-contained Bayesian pipeline (PCA + Gaussian-process emulator + Hamiltonian Monte Carlo)
distilled from the framework behind Ref. [1]. To apply it to a new problem you edit `Input.py`
only; the analyser in `Class.py` stays untouched.

Given experimental data  $\boldsymbol{y}_{\rm exp}$ with covariance $\Sigma_{\rm exp}$ and a
simulator — either an explicit function $y(x)$ or a table of theory predictions
$\{\boldsymbol{y}_{\mathrm{th}}(\boldsymbol{x}_n)\}$ — it samples the posterior of the parameters.

## Layout

```txt
universal_BA/
├── README.md     # this file: formalism -> code map
├── Input.py      # the only file you edit: data files, parameters, priors, HMC settings
├── Class.py      # core: DataLoader / Interpolator / MCMCSampler / OutputAnalyzer
├── Analysis.py   # driver: sample -> save -> summary -> plots
├── plot.py       # standalone joint-posterior plotter
├── Check_emulator.py   # standalone emulator validation (normalized residuals)
├── AutoSciPlot.py, SciPlotStyle.py   # figure style helpers
├── run_analysis.sh   # convenience wrapper around Analysis.py
├── references/   # derivation note (Chinese) behind the analytic potential gradient of §5
└── input/        # data; the shipped example lives in input/Td018/
```

Generated at run time (not tracked by git): `gpr_pca_emulator*.joblib`, `HMC_sample*.dat`,
`HMC_plot*/`, `analysis_run*.log`.

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The shipped example was verified with Python 3.10.18 and the versions pinned in
`requirements.txt` (numpy 2.0.0, scipy 1.15.3, scikit-learn 1.7.0, matplotlib 3.10.3,
seaborn 0.13.2, joblib 1.5.1).

## Quick start

Run everything **from the repository root** — `Input.py` resolves all data paths relative to the
working directory.

```bash
python3 Input.py            # optional: validate the configuration
python3 Analysis.py         # emulator + sampling + summary + plots
python3 plot.py             # optional: redraw the joint distribution only
python3 Check_emulator.py   # optional: validate the PCA+GP emulator itself
```

The repository ships a complete, runnable example (`input/Td018/`, 200 design points, 57
observables, 5 sampled parameters), which is what the active `config` in `Input.py` points at.
A first run from scratch takes about **5 minutes** on a modern laptop: ~18 s to train the five
Gaussian processes, then ~4 min for the 8-chain sampling (8-way parallel, ~0.13 s per HMC step
with the analytic gradient). Later runs reuse the cached emulator and samples and finish in
seconds.

To apply the pipeline to your own problem:

1. Put your data files in a directory under `input/` and point `config.input_dir` at it (see
   [Data-file formats](#data-file-formats)).
2. Edit the rest of `config` in `Input.py`: `N_parameter`, `Bayesian_bound`, `parameter_names`,
   `fixed_params`, the emulator options and the HMC settings.
3. Run `python3 Analysis.py`.

> **Caching caveat.** `Analysis.py` decides purely on *file existence* whether to reuse an
> emulator (`config.model_filepath`) and a sample file (`config.sample_filepath`). If you switch
> datasets, change `N_parameter`, or retune a prior but keep the old output file names, the stale
> emulator or samples will be silently reused. Delete those files, or give each problem its own
> file names — as the shipped example does with its `*_Td018` suffix.

## Example dataset (`input/Td018/`)

| File | Shape | Role |
|---|---|---|
| `bottomonium_param.dat` | 200 × 6 | design $\{x_n\}$: 5 sampled parameters + 1 fixed ($T_d$) |
| `filtered_bottomonium_results.dat` | 200 × 57 | theory table $\{y_{\mathrm{th}}(x_n)\}$ |
| `filtered_exp_data.dat` | 57 | $\boldsymbol{y}_{\mathrm{exp}}$ |
| `filtered_exp_stat.dat` | 57 | diagonal statistical variances |
| `syst_cov_matrix.dat` | 57 × 57 | systematic covariance matrix |

The active `config` reproduces a run at $T_d = 180$ MeV and writes its output under the
`*_Td018` / `HMC_plot_Td018` names, kept separate from the general defaults so that neither the
saved model nor the archived samples are overwritten.

## Configuration (`Input.py`)

### Data files

| Field | Meaning |
|---|---|
| `input_dir` | directory holding all data files; resolved relative to the working directory |
| `parameter_file` | design $\{x_n\}$; one row of `N_parameter + len(fixed_params)` columns per point |
| `theoretical_data_file` | theory table $\{y_{\mathrm{th}}(x_n)\}$; one observable vector per row |
| `experimental_data_file` | $\boldsymbol{y}_{\mathrm{exp}}$ (1D) |
| `experimental_sigma_stat_file` | $\Sigma_{\mathrm{exp}}^{\mathrm{stat}}$: diagonal variances (1D) |
| `experimental_sigma_syst_file` | $\Sigma_{\mathrm{exp}}^{\mathrm{syst}}$: covariance matrix (2D) |

With `analytical=True` neither the design nor the theory table is read, so those two fields are
ignored.

### Parameters, emulator, sampler

| Field | Meaning |
|---|---|
| `N_parameter` | number of parameters to estimate, $d$ |
| `Bayesian_bound` | range of each parameter; length **must equal** `N_parameter` |
| `parameter_names` | name of each parameter (LaTeX), length **must equal** `N_parameter` |
| `fixed_params` | parameters appended at the end of the sampled vector and held fixed (e.g. $T_d$) |
| `analytical` / `analytical_model` | `True`: explicit emulator, zero theoretical error, PCA+GP skipped; define `analytical_model(x)` |
| `n_pca_components` | number of principal components kept (`analytical=False` only) |
| `delta_t`, `leapfrog_steps` | leapfrog step size $\delta_t$ and steps per trajectory $N_t$ |
| `target_number`, `burn_in`, `decorrelation_length` | kept samples per chain / burn-in steps / decorrelation length |
| `n_chains`, `random_seed` | number of parallel chains / RNG seed |
| `gradient_epsilon` | finite-difference step used by `scipy.optimize.approx_fprime` (`analytical=True` branch only) |
| `variance_inflation` | variance inflation factor $f$, Eq. (5) |
| `annealing_factor` | simulated-annealing burn-in strength, Eq. (6); `1.0` disables it |
| `model_filepath` | save/load path of the trained PCA+GP model |
| `sample_filepath`, `chain_filepath` | output paths for the merged / per-chain samples |
| `plot_output_dir` | output directory for all figures |
| `figure_annotation` | annotation stamped on the figures (e.g. `r'$T_d = 180$ MeV'`); `None` disables it |

`config.validate()` raises clear dimension errors if `Bayesian_bound` / `parameter_names`
disagree with `N_parameter`; `Interpolator.fit` checks that the design table has exactly
`N_parameter + len(fixed_params)` columns.

## Formalism ↔ code

The equations are numbered (1)–(18) in order of appearance, and every symbol of every equation is
mapped to the code object that holds it. The formalism itself is the one of Ref. [1].

### 1. Notation

| Symbol | Code |
|---|---|
| $\boldsymbol{x}$ | sampled parameter vector (`current_x`, `x_t`), length $d$ = `config.N_parameter` |
| $\boldsymbol{x}_{\mathrm{full}}$ | `x_full = np.append(x, config.fixed_params)` |
| $\boldsymbol{y}_{\mathrm{exp}}$, $N$ | `data_loader.experimental_data` |
| $\boldsymbol{y}(\boldsymbol{x})$ | first return of `Interpolator.predict(x_full)` |
| $\Delta\boldsymbol{y}$ | `delta_y = mu_x - data_loader.experimental_data` |
| $\Sigma_{\mathrm{exp}}$ | `data_loader.Cov_exp = np.diag(experimental_sigma_stat) + experimental_sigma_syst` |
| $\Sigma_{\mathrm{th}}(\boldsymbol{x})$ | second return of `Interpolator.predict(x_full)` |
| $\Sigma(\boldsymbol{x})$ | `Cov = Cov_x + self.Cov_exp` |
| $V(\boldsymbol{x})$ | `0.5 * chi_square(x)` |
| $f$ | `config.variance_inflation` (`self.variance_inflation`) |
| $\boldsymbol{U},\ \boldsymbol{U}^{\dagger}$ | `pca.components_` ($n_{\mathrm{pc}}\times N$) and its transpose |
| $\overline{\boldsymbol{y}}$ | `pca.mean_` |
| $P_m(\boldsymbol{x}),\ \boldsymbol{P}(\boldsymbol{x})$ | per-PC `gp.predict` mean; `Y_pred_pca` |
| $P_q$ | PC training targets; `gp.y_train_` holds them `normalize_y`-rescaled, undone in `_ensure_gradient_cache` by `* y_std + y_mean` |
| $\sigma^2_{P,m}(\boldsymbol{x})$ | `gp.predict(..., return_std=True)` squared |
| $K_m$ | `gp.kernel_(gp.X_train_)` (+ the $10^{-10}$ ridge `gp.alpha`) |
| $l_m,\ C_m,\ \sigma^2_{\mathrm{wn},m}$ | `gp.kernel_.k1.k2.length_scale`, `gp.kernel_.k1.k1.constant_value`, `gp.kernel_.k2.noise_level` |
| $\boldsymbol{x}_p$ | `gp.X_train_` |
| $\delta_t,\ N_t$ | `config.delta_t`, `config.leapfrog_steps` |

### 2. Posterior and likelihood

$$
\mathcal{P}_{\mathrm{posterior}}(\boldsymbol{x}\mid\boldsymbol{y}_{\mathrm{exp}})
=\frac{\mathcal{P}(\boldsymbol{y}_{\mathrm{exp}}\mid\boldsymbol{y}(\boldsymbol{x}))\,
\mathcal{P}_{\mathrm{prior}}(\boldsymbol{x})}{\mathcal{P}(\boldsymbol{y}_{\mathrm{exp}})}
\tag{1}
$$

$$
\mathcal{P}(\boldsymbol{y}_{\mathrm{exp}}\mid\boldsymbol{y}(\boldsymbol{x}))
=\frac{\exp\left[-\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}(\boldsymbol{x})\Delta\boldsymbol{y}\right]}
{\sqrt{(2\pi)^{N}\det[\Sigma(\boldsymbol{x})]}}
\tag{2}
$$

$$
\Sigma(\boldsymbol{x})=\Sigma_{\mathrm{th}}(\boldsymbol{x})+\Sigma_{\mathrm{exp}},
\qquad
\Delta\boldsymbol{y}=\boldsymbol{y}(\boldsymbol{x})-\boldsymbol{y}_{\mathrm{exp}}
\tag{3}
$$

- The prior is flat inside `config.Bayesian_bound`; it enters through the hard-wall reflection in
  `MCMCSampler.sample` (§6), not as a term in the code.
- The likelihood (2) is evaluated in `MCMCSampler.chi_square`: `mu_x, Cov_x = self.interpolator.predict(x_full)`,
  `delta_y = mu_x - self.data_loader.experimental_data`, `Cov = Cov_x + self.Cov_exp` — i.e. Eq. (3).

### 3. Effective potential and variance inflation

$$
\chi^2(\boldsymbol{x})=\Delta\boldsymbol{y}^{T}\Sigma^{-1}(\boldsymbol{x})\Delta\boldsymbol{y}
+\ln\det\Sigma(\boldsymbol{x}),
\qquad
V(\boldsymbol{x})=\tfrac{1}{2}\chi^2(\boldsymbol{x})
\tag{4}
$$

$$
\chi^2(\boldsymbol{x})=\frac{1}{f}\Delta\boldsymbol{y}^{T}\Sigma^{-1}(\boldsymbol{x})\Delta\boldsymbol{y}
+\ln\det\Sigma(\boldsymbol{x})+N\ln f
\tag{5}
$$

$$
f_i=f_{\text{final}}\,a^{\,1-i/N_{\text{burn}}},\qquad i\le N_{\text{burn}}
\tag{6}
$$

- $\Delta\boldsymbol{y}^{T}\Sigma^{-1}\Delta\boldsymbol{y}$ ↔ `chi_1 = delta_y @ Cov_inv @ delta_y.T`
- $\ln\det\Sigma$ ↔ `chi_2 = np.log(np.linalg.det(Cov))`
- Eq. (5) ↔ `chi_1 / f + chi_2 + d * np.log(f)` in `MCMCSampler.chi_square`, with $f$ ↔ `config.variance_inflation`
- Eq. (4) ↔ `0.5 * self.chi_square(x)`; only $\chi^2$ differences enter the Metropolis test.
- **The $1/f$ multiplies the residual quadratic form only**, never $\ln\det\Sigma$. Dividing
  everything by $f$ would flatten the $\boldsymbol{x}$-dependence of $\ln\det\Sigma$ and distort the
  posterior; the analytic gradient, Eq. (10), keeps the same placement.
- If `analytical=True`, $\Sigma=\Sigma_{\mathrm{exp}}$ is $\boldsymbol{x}$-independent, so the code
  takes the shortcut `(delta_y @ self._cov_inv @ delta_y.T + self._logdet) / f`, which differs from
  Eq. (5) by an $\boldsymbol{x}$-independent constant only.
- Eq. (6) ↔ `self.variance_inflation = self._f_final * self.annealing_factor ** (1.0 - i / self.burn_in)`
  in `MCMCSampler.sample` ($a$ ↔ `config.annealing_factor`, $N_{\text{burn}}$ ↔ `config.burn_in`);
  for $i>N_{\text{burn}}$ the inflation equals $f_{\text{final}}$ exactly, so the sampled posterior
  is untouched. $a=1$ disables it.

### 4. Emulator: PCA + GP

$$
\boldsymbol{y}(\boldsymbol{x})=\boldsymbol{U}^{\dagger}\boldsymbol{P}(\boldsymbol{x})+\overline{\boldsymbol{y}},
\qquad
\Sigma_{\mathrm{th}}(\boldsymbol{x})=2\,\boldsymbol{U}^{\dagger}\,
\mathrm{diag}\big(\sigma^2_{P,1}(\boldsymbol{x}),\sigma^2_{P,2}(\boldsymbol{x}),\dots\big)\boldsymbol{U}
\tag{7}
$$

- $\boldsymbol{U}$ ↔ `pca.components_` (rows = principal directions), so $\boldsymbol{U}^{\dagger}=\boldsymbol{U}^{T}$
  is the decoding matrix; $\overline{\boldsymbol{y}}$ ↔ `pca.mean_`.
- $\boldsymbol{y}(\boldsymbol{x})$ ↔ `Y_pred_original = np.array(Y_pred_pca) @ Unitary + mu` in `Interpolator.predict`.
- $\Sigma_{\mathrm{th}}(\boldsymbol{x})$ ↔ `Interpolator.sigma_th_from_var(Y_std_pca ** 2)`. The factor 2 is the paper
  convention $\Sigma_{\mathrm{th}}=2\Sigma_{\mathrm{Gauss}}$, "the uncertainty of the Schrödinger
  evolution framework"; it appears in the code exactly once, inside `sigma_th_from_var`, and must not
  be dropped — the trace and quadratic terms of Eq. (10) scale with it.
- PCA is trained in `Interpolator.fit` on the theory table: `PCA(n_components=config.n_pca_components)`,
  `Y_pca = pca.fit_transform(y_data)`.

$$
\overline{P}_m(\boldsymbol{x})=\sum_{p,q=1}^{n_{\mathrm{train}}}\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)
\left(K_m^{-1}\right)_{pq}P_q,
\qquad
\sigma^2_{P,m}(\boldsymbol{x})=\kappa_m(\boldsymbol{x},\boldsymbol{x})
-\kappa_m^{T}K_m^{-1}\kappa_m
\tag{8}
$$

$$
\kappa_m(\boldsymbol{x},\boldsymbol{x}')=C_m^2\exp\!\left(-\frac{\|\boldsymbol{x}-\boldsymbol{x}'\|_2^2}{2l_m^2}\right),
\qquad
K_m=\kappa_m(\boldsymbol{x}_p,\boldsymbol{x}_q)+\sigma^2_{\mathrm{wn},m}\delta_{pq}
\tag{9}
$$

- Eq. (8) ↔ `gp.predict([x], return_std=True)`, one `GaussianProcessRegressor` per principal
  component in `Interpolator.gpr_model`.
- $K_m$ ↔ `gp.kernel_(gp.X_train_)`; it **includes the white noise** on the diagonal, as in the paper.
- $l_m,C_m,\sigma^2_{\mathrm{wn},m}$ ↔ `gp.kernel_.k1.k2.length_scale`, `gp.kernel_.k1.k1.constant_value`,
  `gp.kernel_.k2.noise_level`, i.e. the kernel built in `Interpolator.fit`:
  `ConstantKernel * RBF + WhiteKernel`, fitted with `normalize_y=True`.
- $\boldsymbol{x}_p$ ↔ `gp.X_train_`, $n_{\mathrm{train}}$ ↔ `gp.X_train_.shape[0]`.

### 5. Analytic potential gradient

The leapfrog integrator needs $\partial V/\partial x_i$, Eq. (10). For `analytical=False` the code
does **not** finite-difference the potential: `Interpolator.predict_with_grad` + `MCMCSampler.gradient`
evaluate the derivative analytically (one emulator call per gradient instead of $d+1$).

$$
\frac{\partial V}{\partial x_i}
=\frac{1}{f}\left[
\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
-\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
\right]
+\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right)
\tag{10}
$$

| Term | Code (in `MCMCSampler.gradient`) |
|---|---|
| $\Delta\boldsymbol{y}^{T}\Sigma^{-1}\partial\Delta\boldsymbol{y}/\partial x_i$ | `residual = delta_y @ Cov_inv @ dy_dx[:, i]` |
| $\Delta\boldsymbol{y}^{T}\Sigma^{-1}(\partial\Sigma/\partial x_i)\Sigma^{-1}\Delta\boldsymbol{y}$ | `quadratic = delta_y @ Cov_inv @ dCov @ Cov_inv @ delta_y` |
| $\mathrm{tr}(\Sigma^{-1}\partial\Sigma/\partial x_i)$ | `trace = np.trace(Cov_inv @ dCov)` |
| full assembly | `grad[i] = (residual - 0.5*quadratic)/f + 0.5*trace` |

$$
\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
=\boldsymbol{U}^{\dagger}\frac{\partial\boldsymbol{P}}{\partial x_i}
\tag{11}
$$

↔ `dy_dx = Unitary.T @ dP_dx_pca`, shape `(N, d)`, in `Interpolator.predict_with_grad`.

$$
\frac{\partial\overline{P}_m(\boldsymbol{x})}{\partial x_i}
=-\frac{1}{l_m^{2}}\sum_{p=1}^{n_{\mathrm{train}}}(x-x_p)_i\,
\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}P\right)_p
\tag{12}
$$

↔ `v = kappa * K_inv_P`, `dP_dx = -(x * v.sum() - v @ X_train) / l_m**2`, where
`K_inv_P = K_inv @ (P_train - y_mean)` — the de-meaned training targets, precomputed in
`Interpolator._ensure_gradient_cache` (`y_mean` = `gp._y_train_mean`, the `normalize_y` offset) —
and $\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)$ ↔ `gp.kernel_(x[None, :], gp.X_train_)`.

$$
\frac{\partial\sigma^2_{P,m}(\boldsymbol{x})}{\partial x_i}
=\frac{2}{l_m^{2}}\sum_{p=1}^{n_{\mathrm{train}}}(x-x_p)_i\,
\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}\kappa_m(\boldsymbol{x})\right)_p
\tag{13}
$$

↔ `g = kappa * (K_inv @ kappa)`, `dvar_dx = 2 * y_std**2 * (x * g.sum() - g @ X_train) / l_m**2`.
The factor `y_std**2` is `gp._y_train_std ** 2`: `normalize_y=True` makes `predict(..., return_std=True)`
return the variance rescaled to physical units, so the derivative must carry the same rescaling to be
the derivative of the $\Sigma_{\mathrm{th}}$ actually used in $\chi^2$ (it is $1$ if `normalize_y=False`).
The kernel-space term $\kappa_m(\boldsymbol{x},\boldsymbol{x})$ and the white-noise floor are
$\boldsymbol{x}$-independent, so they drop out of Eq. (13).

$$
\frac{\partial\Sigma}{\partial x_i}
=\frac{\partial\Sigma_{\mathrm{th}}}{\partial x_i}
=2\,\boldsymbol{U}^{\dagger}\,\mathrm{diag}\!\left(\frac{\partial\sigma^2_{P,m}}{\partial x_i}\right)\boldsymbol{U}
\tag{14}
$$

↔ `dCov = self.interpolator.sigma_th_from_var(dvar_dx[:, i])` — the same routine that builds
$\Sigma_{\mathrm{th}}$ in Eq. (7), so the factor 2 and the decoding matrix have a single source of truth.

Precomputed once per chain, in `Interpolator._ensure_gradient_cache`: `X_train`, `K_inv`,
`K_inv_P`, `length_scale`, `y_std`. Everything else is assembled per evaluation.

Scope: this path exists for `analytical=False` only. With `analytical=True` the emulator is an
explicit function with no PCA+GP bookkeeping, and `MCMCSampler._potential_gradient` keeps using
`scipy.optimize.approx_fprime` on `0.5 * chi_square` (step `config.gradient_epsilon`).

### 6. Hamiltonian Monte Carlo

$$
H(\boldsymbol{x},\boldsymbol{p})=\frac{\boldsymbol{p}^{T}\boldsymbol{p}}{2m}+V(\boldsymbol{x}),
\qquad
\boldsymbol{p}\sim\mathcal{N}(0,mT)
\tag{15}
$$

$$
\xi<\exp\!\left[-\frac{H(\boldsymbol{x}',\boldsymbol{p}')-H(\boldsymbol{x},\boldsymbol{p})}{T}\right]
\tag{16}
$$

- $m=T=1$ in the code: `p_t = np.random.normal(loc=0, scale=1, size=current_x.shape)` and
  `Hamiltonian_present = 0.5 * (Nx_present + p_initial @ p_initial.T)` with `Nx_present = chi_square(x)`,
  i.e. Eq. (15).
- Eq. (16) ↔ `accept_probability = min(1, np.exp(-(Hamiltonian_new - Hamiltonian_present)))`; a
  fresh momentum is drawn for every trajectory and enters $H_{\text{present}}$ (see the comments in
  `MCMCSampler.sample`).

$$
\hat{p}_i(t+\tfrac{\delta_t}{2})=\hat{p}_i(t)-\tfrac{\delta_t}{2}\frac{\partial V}{\partial\hat{x}_i}(t),
\quad
\hat{x}_i(t+\delta_t)=\hat{x}_i(t)+\delta_t\frac{\hat{p}_i(t+\tfrac{\delta_t}{2})}{m},
\quad
\hat{p}_i(t+\delta_t)=\hat{p}_i(t+\tfrac{\delta_t}{2})-\tfrac{\delta_t}{2}\frac{\partial V}{\partial\hat{x}_i}(t+\delta_t)
\tag{17}
$$

| Leapfrog step, Eq. (17) | Code (in `MCMCSampler.sample`) |
|---|---|
| $\partial V/\partial\hat{x}_i(t)$ | `grad_xt = self._potential_gradient(x_t)` |
| first half kick | `p_t_plus_halfdelta = p_t - (delta_t / 2) * grad_xt` |
| drift | `x_t_plus_delta = x_t + delta_t * p_t_plus_halfdelta` |
| $\partial V/\partial\hat{x}_i(t+\delta_t)$ | `grad_xt_plus_delta = self._potential_gradient(x_t_plus_delta)` |
| second half kick | `p_t_plus_delta = p_t_plus_halfdelta - (delta_t / 2) * grad_xt_plus_delta` |

Boundary handling: the flat prior truncates the posterior at `config.Bayesian_bound`, so a
trajectory crossing a face is reflected (`x -> 2*low - x` or `2*high - x`, momentum component
sign-flipped, repeated until back inside — the `while` loop in `MCMCSampler.sample`). Reflection is
reversible, volume-preserving and exactly energy-conserving. The endpoint is then passed through
`MCMCSampler.wrap_parameters`, a periodic fallback that only ever fires on floating-point edge
cases. `chi_square` never wraps periodically: periodic folding would create $\chi^2$ cliffs at the
faces and, in the finite-difference branch, spurious gradients of order $10^{10}$.

### 7. Which branch does what

| Quantity | `analytical=True` | `analytical=False` |
|---|---|---|
| $\boldsymbol{y}(\boldsymbol{x})$ | `config.analytical_model(x_full)` | PCA + GP, `Interpolator.predict` |
| $\Sigma_{\mathrm{th}}(\boldsymbol{x})$ | $0$ | $2\boldsymbol{U}^{\dagger}\mathrm{diag}(\sigma^2)\boldsymbol{U}$, Eq. (7) |
| $\Sigma(\boldsymbol{x})$ | $\Sigma_{\mathrm{exp}}$, $\boldsymbol{x}$-independent | $\Sigma_{\mathrm{th}}(\boldsymbol{x})+\Sigma_{\mathrm{exp}}$ |
| $\chi^2$ | `(chi_1 + logdet) / f` | `chi_1 / f + chi_2 + d*log(f)`, Eq. (5) |
| $\partial V/\partial x_i$ | `approx_fprime` | analytic, `MCMCSampler.gradient`, Eq. (10) |

## Emulator validation (`Check_emulator.py`)

The emulator is the only part of the pipeline that is a statistical
approximation rather than an exact calculation, so it is checked separately.
A GP predicts a *distribution* of outputs, not a single value, so the test is
not "does the emulator reproduce a full calculation?" but "are the calculations
distributed the way the emulator claims?". This is the validation of
Ref. [2], §4.3.4 and Fig. 4.9: for every principal component $m$ and every
design point $\boldsymbol{x}_n$ held out of the training set, the normalized
residual

$$
z_{m,n}=\frac{P_m^{\mathrm{pred}}(\boldsymbol{x}_n)-P_m^{\mathrm{true}}(\boldsymbol{x}_n)}{\sigma_{P,m}(\boldsymbol{x}_n)}
\sim \mathcal{N}(0,1)
\tag{18}
$$

must follow a standard normal distribution. Here

- $P_m^{\mathrm{pred}}(\boldsymbol{x}_n)$ and $\sigma_{P,m}(\boldsymbol{x}_n)$ are the mean and
  standard deviation of `gp_m.predict(x_n, return_std=True)`. This $\sigma_{P,m}$ is exactly the
  one that enters $\Sigma_{\mathrm{th}}(\boldsymbol{x})$, Eq. (7), so Eq. (18) tests the very
  uncertainty the sampler relies on — including the kernel's white-noise term, which is the GP's
  model of the scatter of the training table;
- $P_m^{\mathrm{true}}(\boldsymbol{x}_n)=\boldsymbol{u}_m\cdot(\boldsymbol{y}_{\mathrm{th}}(\boldsymbol{x}_n)-\overline{\boldsymbol{y}})$
  is the truth projected onto the PCA basis of that fold, $\boldsymbol{u}_m$ being row $m$ of
  $\boldsymbol{U}$ (`pca.components_[m]`).

Because the repository ships no separate validation sample, the held-out points are carved out of
the design table: `N_FOLDS = 5` (default) runs a $k$-fold cross-validation, in which every design
point is tested exactly once by an emulator that never saw it, and pools the residuals of all
folds; `N_FOLDS = 1` falls back to a single train/test split of size `TEST_FRACTION`. Folds are
built by `Interpolator.fit`, i.e. the production PCA+GP code path with the production kernel, so
the check cannot drift away from what `Analysis.py` samples with — at the price of training on
fewer points than the deployed emulator (160 instead of 200), which makes the test conservative.
The remaining settings (`N_BINS`, the split seed, the figure style) sit at the top of the file.

The script prints one row per principal component — mean and standard deviation of $z$, the
fractions $|z|<1$ and $|z|<2$ (ideal 0.683 and 0.954), the largest residual, and a
Kolmogorov–Smirnov test against $\mathcal{N}(0,1)$ — and writes
`Emulator_Normalized_Residuals.png` / `.pdf` (one histogram panel per component with the
$\mathcal{N}(0,1)$ density overlaid) into `config.plot_output_dir`. A histogram narrower than
$\mathcal{N}(0,1)$ means the predictive uncertainty is overestimated, a wider one that it is
underestimated. The plotted range stops at $z=\pm6$ — one extreme outlier would otherwise squeeze
all five histograms into an unreadable spike — and the handful of points beyond it are not drawn
but not hidden either: they are counted in the panel annotation and in the `max|z|` column.
Only meaningful for `analytical=False`: with an analytical emulator $\Sigma_{\mathrm{th}}=0$ and
there is no predictive uncertainty to validate.

## Output

File names come from `config.sample_filepath` / `chain_filepath` / `plot_output_dir`. With the
shipped example configuration these are:

- `HMC_sample_Td018.dat` — merged samples (columns `x_1..x_d, chi^2`);
- `HMC_sample_chains_Td018.dat` — per-chain samples (columns `chain_id, x_1..x_d, chi^2`), used for the
  Markov-chain check;
- `HMC_plot_Td018/Markov_Chains_Check.png` — all chains, first parameter;
- `HMC_plot_Td018/Joint_Distributions.png` — joint posterior (marginals on the diagonal, 2D joint
  distributions below it, medians and 95% CI marked, `config.figure_annotation` stamped on top);
- the terminal prints the median and 68% / 95% credible intervals of every parameter
  (`OutputAnalyzer.report_summary`);
- `Check_emulator.py` writes `Emulator_Normalized_Residuals.png` / `.pdf` (normalized-residual
  histograms, one panel per principal component) into the same directory.

## Notes

- `plot.py` is independent of `Analysis.py`, so it can be re-run alone while tuning the figures.
- HMC settings (`delta_t`, `leapfrog_steps`, `variance_inflation`, `annealing_factor`) are
  problem-specific: retune them when you change data set or emulator. `delta_t` is limited by the
  curvature of $\chi^2$ at the mode (the stiffest region the chain visits); the annealed burn-in,
  Eq. (6), exists to keep chains started anywhere in the prior box away from the high-$\chi^2$ desert.
- Chains always start from uniformly random points over the full prior box
  (`Analysis.py`), never from a separate warm-start box.
- The derivation note under `references/` (`partialV_analytic_derivation.md`) is written in
  Chinese; it documents the analytic potential gradient of §5, including the
  $\Sigma_{\mathrm{th}} = 2\Sigma_{\mathrm{Gauss}}$ convention and the variance-inflation factor
  $f$ as it enters the likelihood.

## Citation

If you use this code in your work, please cite Ref. [1]:

```bibtex
@article{Zheng:2025bdc,
    author  = "Zheng, Shuhan and Chen, Baoyi and Du, Xiaojian and Shi, Shuzhe",
    title   = "{Data-driven analysis for the bottomonium potential in the quark-gluon plasma}",
    eprint  = "2512.11536",
    archivePrefix = "arXiv",
    primaryClass  = "nucl-th",
    doi     = "10.1103/b8n3-yrq1",
    journal = "Phys. Rev. C",
    volume  = "114",
    pages   = "024912",
    year    = "2026"
}
```

## License

Released under the [MIT License](LICENSE).

## References

1. S. Zheng, B. Chen, X. Du, S. Shi, *Data-driven analysis for the bottomonium potential in the
   quark-gluon plasma*, Phys. Rev. C **114**, 024912 (2026),
   [arXiv:2512.11536 [nucl-th]](https://arxiv.org/abs/2512.11536), DOI: [10.1103/b8n3-yrq1](https://doi.org/10.1103/b8n3-yrq1).

2. J. E. Bernhard, *Bayesian parameter estimation for relativistic heavy-ion collisions*,
   Ph.D. thesis, Duke University (2018),
   [arXiv:1804.06469 [nucl-th]](https://arxiv.org/abs/1804.06469) — source of the
   PCA + GP emulator formalism and of the normalized-residual validation of `Check_emulator.py`.
