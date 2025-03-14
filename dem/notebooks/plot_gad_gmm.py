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

n_mixes = 2

configs = [
    [
        "Energy",
        "experiment=gmm_idem",
        f"energy.n_mixes={n_mixes}",
    ],
    [
        "GAD unstitched",
        "experiment=gmm_idem_gad",
        "energy.stitching=False",
        "energy.clip_energy=False",
        f"energy.n_mixes={n_mixes}",
    ],
    [
        "GAD stitched",
        "experiment=gmm_idem_gad",
        "energy.stitching=True",
        "energy.clip_energy=True",
        "energy.clamp_min=-0.4",
        "energy.clamp_max=null",
        f"energy.n_mixes={n_mixes}",
    ],
    [
        "GAD offset",
        "experiment=gmm_idem_gad",
        "energy.stitching=False",
        "energy.clip_energy=True",
        "energy.gad_offset=-400",
        "energy.clamp_min=-10",
        "energy.clamp_max=null",
        f"energy.n_mixes={n_mixes}",
    ],
    [
        "GAD stitched and offset",
        "experiment=gmm_idem_gad",
        "energy.stitching=True",
        "energy.clip_energy=True",
        "energy.gad_offset=1",
        "energy.clamp_min=null",
        "energy.clamp_max=-1",
        f"energy.n_mixes={n_mixes}",
    ],
    # [
    #     "GAD stitched and clipped, T=0.1",
    #     "experiment=gmm_idem_gad",
    #     "energy.stitching=True",
    #     "energy.clip_energy=True",
    #     "energy.gad_offset=50",
    #     "energy.clamp_min=0",
    #     "energy.clamp_max=null",
    #     "energy.temperature=0.1",
    #     f"energy.n_mixes={n_mixes}",
    # ],
    # [
    #     "GAD stitched and clipped, div=1e-12",
    #     "experiment=gmm_idem_gad",
    #     "energy.stitching=True",
    #     "energy.clip_energy=True",
    #     "energy.gad_offset=50",
    #     "energy.clamp_min=0",
    #     "energy.clamp_max=null",
    #     "energy.div_epsilon=1e-12",
    #     f"energy.n_mixes={n_mixes}",
    # ],
    [
        "GAD for DEM",
        "experiment=gmm_idem_gad",
        f"energy.n_mixes={n_mixes}",
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

    # only non-special characters
    plt_name = re.sub(r"[^a-zA-Z0-9]", "", name)

    for plot_style in ["imshow", "contours"]:

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

        fig_name = f"plots/gmmgad{n_mixes}_{plt_name}_{plot_style}.png"
        img2.save(fig_name)
        print(f"Saved {fig_name}")

    # U = lambda x: energy_function.energy(
    #     torch.tensor([x], device=energy_function.device, dtype=torch.float32)
    # )

    # print(f"Energy at [0, 0] = {U([0, 0]).item()}")
    # print(f"Energy at [-1.7, 0] = {U([-1.7, 0]).item()}")
    # print(f"Energy at [1.7, 0] = {U([1.7, 0]).item()}")

    energy_function.plot_hessian_eigenvalues(name=name)
    fig_name = f"plots/gmmgad{n_mixes}_{plt_name}_hessian_eigenvalues.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    # get minima
    minima = energy_function.get_minima()
    # get transition state by taking the middle of the first two minima
    y_idx = 1
    transition_state = (minima[0][y_idx] + minima[1][y_idx]) / 2

    energy_function.plot_energy_crossection(name=name, y_value=transition_state.item())
    fig_name = f"plots/gmmgad{n_mixes}_{plt_name}_crossection.png"
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")

    # deinitialize hydra
    GlobalHydra().clear()
