import torch
import numpy as np
from dem.energies.gmm_pseudoenergy import GMMPseudoEnergy

import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100
# mpl.rcParams['figure.figsize'] = [10, 10]

import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

import re


configs = [
    [
        "name=Force L2, Hessian Tanh Mult",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=10.0",
        "energy.hessian_eigenvalue_penalty=tanh_mult",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        # "energy.forces_norm=2",
        # "energy.force_exponent=2",
    ],
    [
        "name=Force L2",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=0.0",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        # "energy.forces_norm=2",
        # "energy.force_exponent=2",
    ],
]

for config in configs:

    name = config[0]
    overrides = config[1:]
    # Only initialize if not already initialized
    if not GlobalHydra().is_initialized():
        # Initialize hydra with the same config path as train.py
        hydra.initialize(config_path="../../configs", version_base="1.3")
        # Load the experiment config for GMM with pseudo-energy
        cfg = hydra.compose(config_name="train", overrides=overrides)

    # Instantiate the energy function using hydra, similar to train.py
    energy_function = hydra.utils.instantiate(cfg.energy)

    # energy_function.forces_norm = 2
    # energy_function.force_exponent = 2

    # only non-special characters
    plt_name = re.sub(r"[^a-zA-Z0-9]", "", name)

    img = energy_function.get_single_dataset_fig(
        # samples=scipy_saddle_points,
        samples=None,
        name=name,
        plot_gaussian_means=True,
        grid_width_n_points=800,
        use_imshow=True,
        # with_legend=False,
        # plot_prob_kwargs={"cmap": "turbo"},
        # plot_sample_kwargs={"color": "red"},
    )
    plt.tight_layout(pad=0)

    # save image
    fig_name = f"plots/gmm_potential_{plt_name}.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    energy_function.update_transition_states(abs_ev_tol=1e-6, grad_tol=1e0)
    scipy_saddle_points = energy_function.find_saddle_points_scipy()

    img = energy_function.get_single_dataset_fig(
        samples=scipy_saddle_points,
        name=name,
        plot_gaussian_means=False,
        # plot_prob_kwargs={"cmap": "turbo"},
        # plot_sample_kwargs={"color": "red"},
        grid_width_n_points=800,
        use_imshow=True,
        with_legend=False,
    )
    plt.tight_layout(pad=0)

    # save image
    fig_name = f"plots/gmm_potential_saddle_points_{plt_name}.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")
