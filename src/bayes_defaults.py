"""MCMC sampler settings shared by every Bayesian analysis.

All fits use NUTS (via NumPyro) with target acceptance 0.9 and, by default,
5000 draws, 2000 tuning steps, 4 chains and seed 42. ``add_sampling_args``
adds ``--draws`` / ``--tune`` / ``--chains`` / ``--seed`` to a script's CLI;
the target acceptance and the sampler backend can only be changed here.
``fit_bambi`` fits a Bambi model with these settings.
"""

from __future__ import annotations

import argparse

TARGET_ACCEPT = 0.9
INFERENCE_METHOD = "numpyro"
DEFAULT_DRAWS = 5000
DEFAULT_TUNE = 2000
DEFAULT_CHAINS = 4
DEFAULT_SEED = 42


def fit_bambi(
    model,
    *,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    random_seed: int = DEFAULT_SEED,
    **extra,
):
    """Fit a Bambi model with project-standard NUTS settings.

    `target_accept` and `inference_method` are forced from the module
    constants; everything else is forwarded to `model.fit()`.
    """
    return model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        target_accept=TARGET_ACCEPT,
        inference_method=INFERENCE_METHOD,
        **extra,
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add --draws / --tune / --chains / --seed to an argparse parser."""
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
