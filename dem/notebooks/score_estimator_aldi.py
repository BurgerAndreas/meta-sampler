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
import time

from dem.models.components.sde_integration import integrate_sde
from dem.models.components.score_estimator import estimate_grad_Rt
from dem.models.components.sdes import VEReverseSDE

configs = [
    [
        "Energy",
        "experiment=aldi2d_idem",
        # "experiment=dw_idem",
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
        config_name="train", overrides=["experiment=aldi2d_idem_pseudo"] + overrides
    )

    # Instantiate the energy function using hydra, similar to train.py
    energy_function = hydra.utils.instantiate(cfg.energy)
    noise_schedule = hydra.utils.instantiate(cfg.model.noise_schedule)

    # only non-special characters
    plt_name = re.sub(r"[^a-zA-Z0-9]", "", name)
    
    device = energy_function.device
    num_samples = cfg.model.num_buffer_samples_to_generate_init
    num_samples = 8
    print(f"num_samples: {num_samples}")
    gen_batch_size = cfg.model.gen_batch_size
    
    #################################################################################
    #################################################################################
    # compare if estimate_grad_Rt gives the same result always
    print("\n", "-"*40)
    n = 30
    mean_abs_diffs = []
    mean_rel_diffs = []
    # mc_samples = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # mc_samples = [2, 4, 6, 8, 12, 16, 24]
    mc_samples = np.arange(2, 26, 1)
    for b in mc_samples:
        abs_diffs = []
        rel_diffs = []
        for i in range(n):
            x = torch.randn(num_samples, 2, device=device)
            score_with_grad = estimate_grad_Rt(
                t=torch.tensor(0.5, device=device),
                x=x.clone().detach(),
                energy_function=energy_function,
                noise_schedule=noise_schedule,
                num_mc_samples=b,
                temperature=1.0,
            )
            score_again = estimate_grad_Rt(
                t=torch.tensor(0.5, device=device),
                x=x.clone().detach(),
                energy_function=energy_function,
                noise_schedule=noise_schedule,
                num_mc_samples=b,
                temperature=1.0,
            )
            diff = torch.abs(score_with_grad - score_again)
            abs_diffs.append(diff.mean().item())
            rel_diffs.append((diff/score_with_grad).mean().item())
            
        print(f"{b}: Abs: {np.mean(abs_diffs):.3e} Rel: {np.mean(rel_diffs):.3e}")
        mean_abs_diffs.append(np.mean(abs_diffs))
        mean_rel_diffs.append(np.mean(rel_diffs))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot for absolute differences
    ax1.plot(mc_samples, mean_abs_diffs)
    ax1.set_title("Absolute Differences")
    ax1.set_xlabel("Number of MC Samples")
    ax1.set_ylabel("Mean Absolute Difference")
    # ax1.set_xscale('log')
    
    # Second subplot for relative differences
    ax2.plot(mc_samples, mean_rel_diffs)
    ax2.set_title("Relative Differences")
    ax2.set_xlabel("Number of MC Samples")
    ax2.set_ylabel("Mean Relative Difference")
    # ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f"plots/score_estimator_aldi2d_abs_rel_diffs.png")
    plt.close()
    print(f"Saved plot to plots/score_estimator_aldi2d_abs_rel_diffs.png")