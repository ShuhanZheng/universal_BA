# Analysis.py --- Universal Bayesian Analysis: main driver
#
# Run inside universal_BA/:
#   python3 Analysis.py
#
# Pipeline: validate input -> load data -> build emulator (analytical or
# PCA+GP) -> parallel HMC sampling -> save samples -> report statistics ->
# quick Markov-chain check -> joint-distribution plot (external plot.py).

import os
import numpy as np

from Input import config
from Class import DataLoader, Interpolator, OutputAnalyzer


def main():
    # 1. validate the input configuration
    config.validate()
    print("=" * 60)
    print("Universal Bayesian Analyzer")
    print(f"  N_parameter    = {config.N_parameter}")
    print(f"  parameters     = {config.parameter_names}")
    print(f"  bounds         = {config.Bayesian_bound}")
    print(f"  fixed_params   = {config.fixed_params}")
    print(f"  emulator       = {'analytical (no theoretical error)' if config.analytical else 'PCA+GP'}")
    print(f"  iterations     = {config.iterations}")
    print("=" * 60)

    # 2. load the data
    data_loader = DataLoader(config)
    data_loader.load()

    # 3. build the emulator (analytical or PCA+GP)
    interpolator = Interpolator(config)
    if not config.analytical:
        if os.path.exists(config.model_filepath):
            interpolator.load_model(config.model_filepath)
        else:
            interpolator.fit(data_loader.parameters, data_loader.theoretical_data)
            interpolator.save_model(config.model_filepath)

    # 4. sampling (reuse an existing sample file if present)
    analyzer = OutputAnalyzer(interpolator, data_loader, config)
    if os.path.exists(config.sample_filepath):
        print(f"Sample file {config.sample_filepath} found; loading samples...")
        analyzer.load_all_samples(config.sample_filepath)
        analyzer.load_all_chains(config.chain_filepath)
    else:
        print("No sample file found; start parallel HMC sampling...")
        np.random.seed(config.random_seed)
        # Initial positions are always drawn uniformly from the prior box:
        # there is no separate warm-start box, so an initial point outside
        # Bayesian_bound cannot be configured in the first place.
        init_bounds = config.Bayesian_bound
        print(f"  initial positions drawn from: {init_bounds}")
        initial_x0 = np.array([
            [np.random.uniform(low, high) for (low, high) in init_bounds]
            for _ in range(config.n_chains)
        ])
        analyzer.run_parallel_sampling(
            initial_x0=initial_x0,
            delta_t=config.delta_t,
            T=config.leapfrog_steps,
            iterations=config.iterations,
            n_chains=config.n_chains,
        )
        analyzer.save_all_samples(config.sample_filepath)
        analyzer.save_all_chains(config.chain_filepath)

    # 5. parameter estimation summary (median + 68%/95% credible intervals)
    analyzer.report_summary()

    # 6. quick check: all chains, first dimension
    analyzer.plot_markov_chains(dimension=0)

    # 7. joint probability distributions (external plot.py)
    analyzer.plot_joint_distributions()


if __name__ == "__main__":
    main()
