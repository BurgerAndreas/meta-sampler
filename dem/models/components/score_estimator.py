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

    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples, *args, **kwargs):
        bigger_samples = score_estimator(
            t, x, energy_function, noise_schedule, num_mc_samples, *args, **kwargs
        )

        smaller_samples = score_estimator(
            t,
            x,
            energy_function,
            noise_schedule,
            int(num_mc_samples / 2),
            *args,
            **kwargs,
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


# original implementation in DEM codebase, using vmap
def log_expectation_reward_vmap(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
    return_aux_output: bool = False,
    temperature: float = 1.0,
):
    """Computes the log expectation of rewards using Monte Carlo sampling.

    Args:
        t: Time tensor [1]
        x: Position tensor [D]
        energy_function: Energy function to evaluate samples
        noise_schedule: Noise schedule for perturbing samples
        num_mc_samples: Number of Monte Carlo samples S to use
        clipper: Optional clipper for bounding log rewards
        return_aux_output: Whether to return auxiliary output

    Returns:
        Log expectation of rewards averaged over Monte Carlo samples
    """
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)  # [S]
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)  # [S, D]

    # Add noise to positions
    h_t = noise_schedule.h(repeated_t).unsqueeze(1)  # [S, 1]
    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())  # [S, D]

    # Compute log rewards per MC sample [S]
    if return_aux_output:
        log_rewards, aux_output = energy_function(
            samples, temperature=temperature, return_aux_output=True
        )
    else:
        log_rewards = energy_function(samples, temperature=temperature)

    # Clip log rewards if necessary
    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    # Average log rewards over MC samples [1]
    reward_val = torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)
    if return_aux_output:
        return reward_val, aux_output
    else:
        return reward_val


# Andreas' implementation using batching
def log_expectation_reward_batched(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
    return_aux_output: bool = False,
    temperature: float = 1.0,
):
    """Computes the log expectation of rewards using Monte Carlo sampling.

    Args:
        t: Time tensor [B]
        x: Position tensor [B, D]
        energy_function: Energy function to evaluate samples
        noise_schedule: Noise schedule for perturbing samples
        num_mc_samples: Number of Monte Carlo samples S to use
        clipper: Optional clipper for bounding log rewards
        return_aux_output: Whether to return auxiliary output

    Returns:
        Log expectation of rewards averaged over Monte Carlo samples
    """
    repeated_t = t.unsqueeze(1).repeat_interleave(num_mc_samples, dim=1)  # [B, S]
    repeated_x = x.unsqueeze(1).repeat_interleave(num_mc_samples, dim=1)  # [B, S, D]

    # Add noise to positions
    h_t = noise_schedule.h(repeated_t).unsqueeze(-1)  # [B, S, 1]
    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())  # [B, S, D]

    # Compute log rewards per MC sample [B, S]
    samples = samples.view(-1, samples.shape[-1])  # [B * S, D]
    if return_aux_output:
        log_rewards, aux_output = energy_function(
            samples, temperature=temperature, return_aux_output=True
        )
    else:
        log_rewards = energy_function(samples, temperature=temperature)
    log_rewards = log_rewards.view(t.shape[0], -1)  # [B*S] -> [B, S]

    # Clip log rewards if necessary
    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    # Average log rewards over MC samples [B]
    reward_val = torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)

    if return_aux_output:
        return reward_val, aux_output
    else:
        return reward_val


# Andreas' implementation using batching and torch.autograd.grad
def _estimate_grad_Rt_batched(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    use_vmap: bool = True,
    return_aux_output: bool = False,
    temperature: float = 1.0,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    # computes grad w.r.t. to x
    x = x.requires_grad_(True)
    reward, aux_output = log_expectation_reward_batched(
        t,
        x,
        energy_function,
        noise_schedule,
        num_mc_samples,
        return_aux_output=return_aux_output,
        temperature=temperature,
    )
    grad_output = torch.autograd.grad(
        outputs=reward,
        inputs=x,
        grad_outputs=torch.ones_like(reward),
        create_graph=True,
        # retain_graph=True,
    )[0]
    return grad_output, aux_output


# original implementation in DEM codebase, using vmap and torch.func.grad
def _estimate_grad_Rt_vmap(
    t: torch.Tensor,  # [num_samples_from_buffer]
    x: torch.Tensor,  # [num_samples_from_buffer, D]
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    use_vmap: bool = True,
    return_aux_output: bool = False,
    temperature: float = 1.0,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    def _log_expectation_reward(t, x, energy_function, noise_schedule, num_mc_samples):
        # t: [], x: [D]
        return log_expectation_reward_vmap(
            t,
            x,
            energy_function,
            noise_schedule,
            num_mc_samples,
            return_aux_output=return_aux_output,
            temperature=temperature,
        )

    # argnums=1 -> computes grad w.r.t. to x (zero indexed)
    grad_fxn = torch.func.grad(
        _log_expectation_reward, argnums=1, has_aux=return_aux_output
    )
    # we have two "batch" dimensions: S=num_mc_samples and B=num_samples_from_buffer
    # [B], [B,D] -> [], [D]
    vmapped_fxn = torch.vmap(
        grad_fxn, in_dims=(0, 0, None, None, None), randomness="different"
    )
    # grad_output, aux_output = vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)
    # return grad_output, aux_output
    return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)


def estimate_grad_Rt(
    t,
    x,
    energy_function,
    noise_schedule,
    num_mc_samples,
    use_vmap: bool = True,
    *args,
    **kwargs,
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
        use_vmap: Whether to use vectorized mapping
        return_aux_output: Whether to return auxiliary output
    Returns:
        Gradient of reward function with respect to position
    """
    if use_vmap:
        return _estimate_grad_Rt_vmap(
            t, x, energy_function, noise_schedule, num_mc_samples, *args, **kwargs
        )
    else:
        return _estimate_grad_Rt_batched(
            t, x, energy_function, noise_schedule, num_mc_samples, *args, **kwargs
        )
        # raise NotImplementedError("Non-vmap gradient estimation not implemented")


if __name__ == "__main__":
    from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    # Create a simple test for estimate_grad_Rt
    print("Testing estimate_grad_Rt function...")

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    dim = 2
    num_mc_samples = 100

    # Create energy function (Double Well)
    energy_function = DoubleWellEnergy(device=device, dimensionality=dim)

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # Test with vmap=True
    print(f"Running with vmap=True, batch_size={batch_size}, dim={dim}")
    grad_output, aux_output = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=True,
        return_aux_output=True,
    )
    print(f"Gradient shape: {grad_output.shape}")
    print(
        f"Gradient mean: {grad_output.mean().item():.4f}, std: {grad_output.std().item():.4f}"
    )
    print(f"Auxiliary output keys: {aux_output.keys()}")

    # Test with vmap=False
    print(f"\nRunning with vmap=False, batch_size={batch_size}, dim={dim}")
    energy_function = DoubleWellEnergy(
        device=device, dimensionality=dim, use_vmap=False
    )
    grad_output_no_vmap, aux_output_no_vmap = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=False,
        return_aux_output=True,
    )
    print(f"Gradient shape: {grad_output_no_vmap.shape}")
    print(
        f"Gradient mean: {grad_output_no_vmap.mean().item():.4f}, std: {grad_output_no_vmap.std().item():.4f}"
    )
    print(f"Auxiliary output keys: {aux_output_no_vmap.keys()}")
