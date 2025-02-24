import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def wrap_for_richardsons(score_estimator):
    """Wraps a score estimator function to apply Richardson's extrapolation.

    Args:
        score_estimator: Function that estimates scores given time, position, energy function,
            noise schedule and number of MC samples.

    Returns:
        Function that applies Richardson's extrapolation to improve score estimates by combining
        results from different sample sizes.
    """
    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
        bigger_samples = score_estimator(t, x, energy_function, noise_schedule, num_mc_samples)

        smaller_samples = score_estimator(
            t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    """Computes the log expectation of rewards using Monte Carlo sampling.

    Args:
        t: Time tensor
        x: Position tensor
        energy_function: Energy function to evaluate samples
        noise_schedule: Noise schedule for perturbing samples
        num_mc_samples: Number of Monte Carlo samples to use
        clipper: Optional clipper for bounding log rewards

    Returns:
        Log expectation of rewards averaged over Monte Carlo samples
    """
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    # Add noise to positions
    h_t = noise_schedule.h(repeated_t).unsqueeze(1)
    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

    # Compute log rewards
    log_rewards = energy_function(samples)

    # Clip log rewards if necessary
    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    # Average log rewards
    return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)


def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    use_vmap: bool = True,
):
    """Estimates the gradient of the reward function with respect to position.
    
    This is not directly computing ∇E (gradient of energy), but rather computing:
    \[ \nabla_x \log \mathbb{E}[\exp(E(x + \sqrt{h(t)}\epsilon))] \]

    Uses automatic differentiation and vectorized mapping to compute gradients efficiently.

    Args:
        t: Time tensor
        x: Position tensor
        energy_function: Energy function to evaluate samples
        noise_schedule: Noise schedule for perturbing samples
        num_mc_samples: Number of Monte Carlo samples to use

    Returns:
        Gradient of reward function with respect to position
    """
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    if use_vmap:
        grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
        vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None), randomness="different")
        return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)
    else:
        raise NotImplementedError("Non-vmap gradient estimation not implemented")
        # func must return a single-element Tensor
        return torch.func.grad(
            func=log_expectation_reward, argnums=1
            )(
                t, x, energy_function, noise_schedule, num_mc_samples
            )
