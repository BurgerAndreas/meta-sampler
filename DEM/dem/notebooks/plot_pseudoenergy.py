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

# Only initialize if not already initialized
if not GlobalHydra().is_initialized():
    # Initialize hydra with the same config path as train.py
    hydra.initialize(config_path="../../configs", version_base="1.3")
    # Load the experiment config for GMM with pseudo-energy
    cfg = hydra.compose(config_name="train", overrides=[
        "experiment=gmm_idem_pseudo", 
        "energy.hessian_weight=0.0",
        "energy.energy_weight=1.0",
        "energy.force_weight=0.0",
    ])

# Instantiate the energy function using hydra, similar to train.py
energy_function = hydra.utils.instantiate(cfg.energy)


img = energy_function.get_single_dataset_fig(
    # samples=scipy_saddle_points, 
    samples=None,
    name="Psuedopotential",
    plot_gaussian_means=True,
)

# save image
fig_name = "gmm_potential.png"
plt.savefig(fig_name)
print(f"Saved {fig_name}")