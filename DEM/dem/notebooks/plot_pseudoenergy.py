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
        "Energy",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=0.0",
        "energy.energy_weight=1.0",
        "energy.force_weight=0.0",
    ],
    [
        "Hessian and",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=1.0",
        "energy.hessian_eigenvalue_penalty=and",
        "energy.energy_weight=0.0",
        "energy.force_weight=0.0",
    ],
    [
        "Force L2 Tanh",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=0.0",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        "energy.force_activation=tanh",
    ],
    [
        "Force L2 Tanh AND Hessian",
        "experiment=gmm_idem_pseudo",
        "energy.hessian_weight=1.0",
        "energy.hessian_eigenvalue_penalty=and",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        "energy.force_activation=tanh",
        "energy.term_aggr=1mmultfh",
    ],
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
        # Get both images
        img1 = energy_function.get_single_dataset_fig(
            samples=None,
            name=name,
            plot_gaussian_means=False, 
            grid_width_n_points=800,
            plot_style=plot_style,
            plot_sample_kwargs={"color": "m", "marker": "."},
            colorbar=True,
        )
        
        # save individual images
        img1.save(f"plots/gmm_{plt_name}_{plot_style}.png")
        print(f"Saved {f'plots/gmm_{plt_name}_{plot_style}.png'}")

        img2 = energy_function.get_single_dataset_fig(
            samples=scipy_saddle_points,
            name=name,
            plot_gaussian_means=True,
            grid_width_n_points=800,
            plot_style=plot_style,
            plot_sample_kwargs={"color": "m", "marker": "."},
            colorbar=True,
        )
        
        # # save individual images
        # img2.save(f"plots/gmm_saddles_{plt_name}_{plot_style}.png")

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display the PIL images
        ax1.imshow(img1)
        ax1.axis('off')
        # ax1.set_title(name)
        
        ax2.imshow(img2) 
        ax2.axis('off')
        # ax2.set_title("With Saddle Points")

        plt.tight_layout(pad=0.05)

        # Save combined figure
        fig_name = f"plots/gmm2_{plt_name}_{plot_style}.png"
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")
        plt.close()
        
        for temp in [1.0, 300.0, 3000.0]:
            _name = f"{name} Boltzmann T={temp}"
            energy_function.kbT = temp
            plt.cla(), plt.clf(), plt.close()
            # plot Boltzmann distribution
            img1 = energy_function.get_single_dataset_fig(
                samples=None,
                name=_name,
                plot_gaussian_means=False, 
                grid_width_n_points=800,
                plot_style=plot_style,
                do_exp=True,
                plot_sample_kwargs={"color": "m", "marker": "."},
                colorbar=True,
            )
            
            # save individual images
            fname = f"plots/gmmB{temp}_{plt_name}_{plot_style}.png"
            img1.save(fname)
            print(f"Saved {fname}")

    # deinitialize hydra
    GlobalHydra().clear()
