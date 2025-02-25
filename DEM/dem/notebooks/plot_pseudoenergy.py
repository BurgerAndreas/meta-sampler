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
    # [
    #     "Energy",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=0.0",
    #     "energy.energy_weight=1.0",
    #     "energy.force_weight=0.0",
    # ],
    # [
    #     "Force L2, Hessian Tanh Mult",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=10.0",
    #     "energy.hessian_eigenvalue_penalty=tanh_mult",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=1.0",
    # ],
    # [
    #     "Multiply: Force L2, Hessian Tanh Mult",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=10.0",
    #     "energy.hessian_eigenvalue_penalty=tanh_mult",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=1.0",
    #     "energy.term_aggr=multfh",
    # ],
    # [
    #     "Energy, Force L2, Hessian Tanh Mult",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=10.0",
    #     "energy.hessian_eigenvalue_penalty=tanh_mult",
    #     "energy.energy_weight=1.0",
    #     "energy.force_weight=1.0",
    # ],
    # [
    #     "Force L2",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=0.0",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=1.0",
    # ],
    # [
    #     "Force L2 squared",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=0.0",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=1.0",
    #     "energy.force_exponent=2",
    # ],
    [
        "Force L2 inv",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=0.0",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        "energy.force_exponent=-1",
    ],
    # [
    #     "Hessian Tanh Mult",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=1.0",
    #     "energy.hessian_eigenvalue_penalty=tanh_mult",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=0.0",
    # ],
    # [
    #     "Hessian Mult",
    #     "experiment=gmm_idem_pseudo",
    #     "energy.hessian_weight=1.0",
    #     "energy.hessian_eigenvalue_penalty=mult",
    #     "energy.energy_weight=0.0",
    #     "energy.force_weight=0.0",
    # ],
]

for config in configs:
    print("\n", config[0])

    name = config[0]
    overrides = config[1:]
    # Initialize hydra with the same config path as train.py
    # if not GlobalHydra().is_initialized():
    hydra.initialize(config_path="../../configs", version_base="1.3")
    # Load the experiment config for GMM with pseudo-energy
    cfg = hydra.compose(config_name="train", overrides=overrides)

    # Instantiate the energy function using hydra, similar to train.py
    energy_function = hydra.utils.instantiate(cfg.energy)

    # energy_function.forces_norm = 2
    # energy_function.force_exponent = 2

    # only non-special characters
    plt_name = re.sub(r"[^a-zA-Z0-9]", "", name)
    
    scipy_saddle_points = energy_function.get_true_transition_states()

    for plot_style in ["imshow"]:
        img = energy_function.get_single_dataset_fig(
            # samples=scipy_saddle_points,
            samples=None,
            name=name,
            plot_gaussian_means=False,
            grid_width_n_points=800,
            plot_style=plot_style,
            # with_legend=False,
            # plot_prob_kwargs={"cmap": "turbo"},
            plot_sample_kwargs={"color": "m", "marker": "."},
        )
        plt.tight_layout(pad=0.05)

        # save image
        fig_name = f"plots/gmm_{plt_name}_{plot_style}.png"
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")
        plt.close()
        
        
        img = energy_function.get_single_dataset_fig(
            samples=scipy_saddle_points,
            # samples=None,
            name=name,
            plot_gaussian_means=True,
            grid_width_n_points=800,
            plot_style=plot_style,
            # with_legend=False,
            # plot_prob_kwargs={"cmap": "turbo"},
            plot_sample_kwargs={"color": "m", "marker": "."},
        )
        plt.tight_layout(pad=0.05)

        # save image
        fig_name = f"plots/gmm_saddles_{plt_name}_{plot_style}.png"
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")
        plt.close()

    # deinitialize hydra
    GlobalHydra().clear()
