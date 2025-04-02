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
    gen_batch_size = cfg.model.gen_batch_size

    #################################################################################
    # test 10 forward passes
    n = 10
    t1 = time.time()
    for i in range(n):
        x = torch.randn(num_samples, 2, device=device)
        energy_function(x)
    t2 = time.time()
    print(f"Time per forward pass: {(t2 - t1)/n:.3f} seconds")
    print(
        f"Estimated time for 1000 integration steps: {(t2 - t1)/n * 1000:.3f} seconds"
    )

    #################################################################################
    # test 10 forward passes without gradient
    n = 10
    t1 = time.time()
    with torch.no_grad():
        for i in range(n):
            x = torch.randn(num_samples, 2, device=device)
            energy_function(x)
    t2 = time.time()
    print(f"Time per forward pass: {(t2 - t1)/n:.3f} seconds")
    print(
        f"Estimated time for 1000 integration steps: {(t2 - t1)/n * 1000:.3f} seconds"
    )

    #################################################################################
    # compare if estimate_grad_Rt gives the same result always
    print("\n", "-" * 40)
    n = 10
    t1 = time.time()
    for i in range(n):
        x = torch.randn(num_samples, 2, device=device)
        score_with_grad = estimate_grad_Rt(
            t=torch.tensor(0.5, device=device),
            x=x.clone().detach(),
            energy_function=energy_function,
            noise_schedule=noise_schedule,
            num_mc_samples=cfg.model.num_estimator_mc_samples,
            temperature=1.0,
        )
        score_again = estimate_grad_Rt(
            t=torch.tensor(0.5, device=device),
            x=x.clone().detach(),
            energy_function=energy_function,
            noise_schedule=noise_schedule,
            num_mc_samples=cfg.model.num_estimator_mc_samples,
            temperature=1.0,
        )
        diff = score_with_grad - score_again
        print(f"Difference: {diff.norm():.3e}")
        print(f"Relative difference: {(diff/score_with_grad).norm():.3e}")

    #################################################################################
    # compare if estimate_grad_Rt gives the same result regardless of no_grad
    print("\n", "-" * 40)
    n = 10
    t1 = time.time()
    for i in range(n):
        x = torch.randn(num_samples, 2, device=device)
        score_with_grad = estimate_grad_Rt(
            t=torch.tensor(0.5, device=device),
            x=x.clone().detach(),
            energy_function=energy_function,
            noise_schedule=noise_schedule,
            num_mc_samples=cfg.model.num_estimator_mc_samples,
            temperature=1.0,
        )
        with torch.no_grad():
            score_no_grad = estimate_grad_Rt(
                t=torch.tensor(0.5, device=device),
                x=x.clone().detach(),
                energy_function=energy_function,
                noise_schedule=noise_schedule,
                num_mc_samples=cfg.model.num_estimator_mc_samples,
                temperature=1.0,
            )
        diff = score_with_grad - score_no_grad
        print(f"Difference: {diff.norm():.3e}")
        print(f"Relative difference: {(diff/score_with_grad).norm():.3e}")

    #################################################################################
    # test 10 estimate_grad_Rt
    print("\n", "-" * 40)
    n = 10
    t1 = time.time()
    for i in range(n):
        x = torch.randn(num_samples, 2, device=device)
        estimate_grad_Rt(
            t=torch.tensor(0.5, device=device),
            x=x,
            energy_function=energy_function,
            noise_schedule=noise_schedule,
            num_mc_samples=cfg.model.num_estimator_mc_samples,
            temperature=1.0,
        )
    t2 = time.time()
    print(f"Time per estimate_grad_Rt: {(t2 - t1)/n:.3f} seconds")
    seconds = (t2 - t1) / n * 1000
    print(
        f"Estimated time for 1000 integration steps: {seconds:.3f} seconds ({seconds/60:.3f} minutes)"
    )

    #################################################################################
    t1 = time.time()

    def _grad_fxn(t, x):
        return estimate_grad_Rt(
            t,
            x,
            energy_function,
            noise_schedule,
            cfg.model.num_estimator_mc_samples,
            # use_vmap=self.use_vmap,
            temperature=1.0,
        )

    reverse_sde = VEReverseSDE(_grad_fxn, noise_schedule)
    init_states = integrate_sde(
        sde=reverse_sde,
        x0=torch.randn(num_samples, 2, device=device),
        num_integration_steps=cfg.model.num_integration_steps,
        energy_function=energy_function,
        diffusion_scale=cfg.model.diffusion_scale,
        temperature=1.0,
        batch_size=cfg.model.gen_batch_size,
        no_grad=True,
    ).detach()

    t2 = time.time()
    print(f"Time for integration: {t2 - t1:.1f} seconds")
