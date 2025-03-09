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
        "experiment=dw4_idem",
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

    img1 = energy_function.get_dataset_fig(
        samples=None,
        random_samples=False,
        # name=name,
        # plot_minima=False,
        # grid_width_n_points=800,
        # plot_style=plot_style,
        # plot_sample_kwargs={"color": "m", "marker": "."},
        # colorbar=True,
    )

    img1.save(f"plots/dw4_{plt_name}.png")
    print(f"Saved {f'plots/dw4_{plt_name}.png'}")

    # deinitialize hydra
    GlobalHydra().clear()
