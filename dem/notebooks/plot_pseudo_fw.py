import torch
import numpy as np

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
        "experiment=fw_idem",
    ],
    [
        "|grad| stitched -l1*l2",
        "energy.hessian_weight=1.0",
        "energy.hessian_eigenvalue_penalty=null",
        "energy.energy_weight=0.0",
        "energy.force_weight=1.0",
        "energy.force_activation=null",
        "energy.term_aggr=cond_force",
        "energy.force_scale=1.0",
    ],
    [
        "Pseudopotential for DEM",
    ],
    [
        "|F| for DEM",
        "experiment=fw_idem_condforce",
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
    cfg = hydra.compose(
        config_name="train", overrides=["experiment=fw_idem_pseudo"] + overrides
    )

    # Instantiate the energy function using hydra, similar to train.py
    energy_function = hydra.utils.instantiate(cfg.energy)

    # only non-special characters
    plt_name = re.sub(r"[^a-zA-Z0-9]", "", name)

    for plot_style in ["imshow", "contours"]:
        plt.close()
        # img1 = energy_function.get_dataset_fig(
        #     samples=None,
        #     random_samples=False,
        #     # name=name,
        #     # plot_minima=False,
        #     # grid_width_n_points=800,
        #     # plot_style=plot_style,
        #     # plot_sample_kwargs={"color": "m", "marker": "."},
        #     # colorbar=True,
        # )

        # img1.save(f"plots/dw_{plt_name}_{plot_style}.png")
        # print(f"Saved {f'plots/dw_{plt_name}_{plot_style}.png'}")

        img2 = energy_function.get_single_dataset_fig(
            samples=None,
            name=name,
            # plot_minima=False,
            grid_width_n_points=800,
            plot_style=plot_style,
            # plot_sample_kwargs={"color": "m", "marker": "."},
            colorbar=True,
            quantity="e",
        )

        fig_name = f"plots/fw_{plt_name}_{plot_style}.png"
        img2.save(fig_name)
        print(f"Saved {fig_name}")

    def U(x):
        return energy_function.energy(
            torch.tensor([x], device=energy_function.device, dtype=torch.float32)
        )

    # Plot contour with samples, minima, and transition state
    # Get samples from the energy function
    samples = energy_function.setup_test_set()
    # Get minima and transition states
    minima = energy_function.get_minima()
    transition_states = energy_function.get_true_transition_states()
    # Create the contour plot with samples
    img = energy_function.get_single_dataset_fig(
        samples=samples,
        name=f"{name} with samples and critical points",
        plot_minima=True,
        grid_width_n_points=800,
        plot_style="contours",
        # plot_sample_kwargs={"color": "m", "marker": ".", "alpha": 0.5, "s": 10},
        colorbar=True,
        quantity="e",
        return_fig=False,
    )
    # Save the figure
    fig_name = f"plots/fw_{plt_name}_contour_with_samples.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    energy_function.plot_hessian_eigenvalues(name=name)
    fig_name = f"plots/fw_{plt_name}_hessian_eigenvalues.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    # energy_function.plot_energy_crossection(name=name, y_value=-1.3710, plotting_bounds=(1.3, -1.4))
    energy_function.plot_energy_crossection(
        name=name, 
        #y_value=-1.3710, plotting_bounds=(1.3, -1.4)
    )
    fig_name = f"plots/fw_{plt_name}_crossection.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")
    energy_function.plot_energy_crossection_along_axis(
        name=name, axis=0, axis_value=0, plotting_bounds=(1.3, -1.4)
    )
    fig_name = f"plots/fw_{plt_name}_crossection0.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")
    # energy_function.plot_energy_crossection_along_axis(
    #     name=name, axis=1, axis_value=0, plotting_bounds=(1.3, -1.4)
    # )
    # fig_name = f"plots/dw_{plt_name}_crossection1.png"
    # plt.savefig(fig_name)
    # print(f"Saved {fig_name}")

    energy_function.plot_gradient(name=name)
    fig_name = f"plots/fw_{plt_name}_gradient.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    # deinitialize hydra
    GlobalHydra().clear()

# Just do once for the final config


# energy_function.plot_energy_crossection(
#     name=name, 
#     #y_value=-1.3710, plotting_bounds=(1.3, -1.4)
# )
# fig_name = f"plots/fw_{plt_name}_crossection.png"
# plt.savefig(fig_name)
# print(f"Saved {fig_name}")
