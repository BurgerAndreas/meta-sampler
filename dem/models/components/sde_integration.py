from contextlib import contextmanager

import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function, temperature=1.0):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(
            torch.sum(energy_function(x, temperature=temperature)), x
        )[0].detach()


# TODO: is this docstring really what this is?
def negative_time_descent(
    x, energy_function, num_steps, dt=1e-4, clipper=None, temperature=1.0
):
    """Perform gradient descent in the energy landscape (negative time evolution).

    While the regular diffusion process moves from t=1 to t=0, this function continues
    into "negative time" (t<0) by following the gradient of the energy function.
    This allows exploration of energy landscapes and can help find transition states,
    local minima, or other critical points in the energy surface.

    Unlike the SDE integration which balances drift and diffusion, this performs
    pure gradient descent without stochastic noise.

    Args:
        x: Initial state (typically the final state from regular SDE integration at t=0)
        energy_function: Energy function defining the landscape to explore
        num_steps: Number of gradient descent steps to perform
        dt: Step size for gradient descent
        clipper: Optional clipper to stabilize gradient estimates

    Returns:
        torch.Tensor: Trajectory of samples during negative time evolution [num_steps, batch_size, dimensions]
    """
    samples = []
    for _ in range(num_steps):
        # Compute the gradient of the energy function at the current state
        # This points in the direction of increasing energy
        drift = grad_E(x, energy_function, temperature=temperature)

        # Optionally clip the gradients for numerical stability
        if clipper is not None:
            drift = clipper.clip_scores(drift)

        # Update the state by moving in the direction of the gradient (uphill)
        # This is opposite of traditional gradient descent which minimizes a function
        # Here we're exploring the landscape by potentially climbing energy barriers
        x = x + drift * dt

        # For molecular systems, enforce center of mass conservation
        if energy_function.is_molecule:
            x = remove_mean(
                x, energy_function.n_particles, energy_function.n_spatial_dim
            )

        samples.append(x)

    # Return the full trajectory of states during negative time evolution
    return torch.stack(samples)


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0
):
    """Perform a single step of Euler-Maruyama numerical integration for an SDE.

    The Euler-Maruyama method is a numerical technique for solving stochastic differential
    equations of the form: dx = f(x,t)dt + g(x,t)dW, where dW is a Wiener process (Brownian motion).

    For an SDE: dx = f(x,t)dt + g(x,t)dW,
    Each step computes: x_{n+1} = x_n + f(x_n,t_n)Δt + g(x_n,t_n)√(Δt)ξ, where ξ ~ N(0,1)

    Args:
        sde: The stochastic differential equation providing drift f(x,t) and diffusion g(x,t) terms
        t: Current time point for evaluating the SDE
        x: Current state of the system
        dt: Time step size for numerical integration
        diffusion_scale: Scaling factor for the diffusion term (stochastic noise component)

    Returns:
        tuple: (Updated state x_next, Drift term)
    """
    # Calculate drift term: f(x,t)Δt
    # The drift represents the deterministic part of the SDE
    drift = sde.f(t, x) * dt

    # Calculate diffusion term: g(x,t)√(Δt)ξ
    # The diffusion represents the stochastic part of the SDE
    # - sde.g(t, x): diffusion coefficient at current state and time
    # - np.sqrt(dt): scaling for Brownian motion over time step dt
    # - torch.randn_like(x): random normal noise (representing the Wiener increment)
    # - diffusion_scale: user-specified scaling factor for the noise
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Update the state according to the Euler-Maruyama method
    # x_{n+1} = x_n + drift + diffusion
    x_next = x + drift + diffusion

    return x_next, drift


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
    temperature=1.0,
):
    """Numerically integrate a stochastic differential equation (SDE).

    This function implements the Euler-Maruyama numerical integration scheme for SDEs,
    which is a stochastic extension of the Euler method for ODEs. It transforms samples
    between the prior noise distribution and the target data distribution.

    Args:
        sde: The stochastic differential equation to integrate
        x0: Initial samples to start the integration from
        num_integration_steps: Number of discretization steps for numerical integration
        energy_function: Energy function defining the target distribution
        reverse_time: If True, integrate from t=1 to t=0 (noise→data); if False, t=0 to t=1 (data→noise)
        diffusion_scale: Scaling factor for the noise/diffusion term
        no_grad: If True, disable gradient computation during integration
        time_range: Total time range for integration
        negative_time: If True, continue integration into negative time after reaching t=0
        num_negative_time_steps: Number of steps for negative time integration
        clipper: Optional clipper to stabilize score/gradient estimates

    Returns:
        torch.Tensor: Trajectory of samples during integration [num_steps, batch_size, dimensions]
    """
    # Set up the time discretization based on direction of integration
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    # Create evenly spaced time points for integration
    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x0.device
    )[
        :-1
    ]  # Remove the last point as it's handled separately

    # Initialize the current state with the input samples
    x = x0
    samples = []

    # Conditionally disable gradient computation to save memory during inference
    with conditional_no_grad(no_grad):
        # Main integration loop: step through time points
        for t in times:
            # Perform a single Euler-Maruyama step of the SDE
            # This updates the state x based on the drift and diffusion terms
            x, f = euler_maruyama_step(
                sde, t, x, time_range / num_integration_steps, diffusion_scale
            )

            # For molecular systems, enforce center of mass conservation
            # by removing the mean displacement (fixing translation invariance)
            if energy_function.is_molecule:
                x = remove_mean(
                    x, energy_function.n_particles, energy_function.n_spatial_dim
                )

            # Save the current state in the trajectory
            samples.append(x)

    # Combine all states into a single tensor representing the full trajectory
    samples = torch.stack(samples)

    try:
        assert torch.isfinite(samples).all()
    except Exception as e:
        nan_idx = torch.isnan(samples[-1]).nonzero()
        print(f"Error in integrate_sde: {e}")
        print(f"nan_idx numel: {nan_idx.numel()}")
        print(f"x0: {x0[nan_idx].shape}")
        for i, s in enumerate(samples):
            print(f"t: {times[i]}. x: \n{s[nan_idx]}")
        # write to file
        with open("nan_samples.txt", "w") as f:
            f.write(f"nan_idx numel: {nan_idx.numel()}")
            f.write(f"x0: {x0[nan_idx].shape}")
            for i, s in enumerate(samples):
                f.write(f"\nt: {times[i]}. x: \n{s[nan_idx]}")
        print("nan samples written to nan_samples.txt")
        raise e

    # TODO: is this really what this is?
    # Optional: Continue integration into negative time (past t=0)
    # This is useful for finding transition states or exploring energy landscapes
    if negative_time:
        print("doing negative time descent...")
        # Perform gradient descent in the energy landscape
        # This follows the gradient of the energy function (not the score function)
        samples_langevin = negative_time_descent(
            x,
            energy_function,
            num_steps=num_negative_time_steps,
            clipper=clipper,
            temperature=temperature,
        )
        # Concatenate the negative time trajectory with the regular trajectory
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples


######################################################################################################
# Kirill's mode generation
######################################################################################################


def constrained_euler_maruyama_step(
    sde: VEReverseSDE,
    t: torch.Tensor,
    x: torch.Tensor,
    dt: float,
    constant_of_motion_fn,
    diffusion_scale=1.0,
):
    """Perform a constrained Euler-Maruyama step that preserves a constant of motion.

    This extends the standard Euler-Maruyama method by projecting the increment to
    ensure that a specified function (constant of motion) remains unchanged during integration.

    The projection formula is:
    dx_projected = dx - (dt*∂c/∂t + <∂c/∂x, dx>) * ∂c/∂x / ||∂c/∂x||²

    Args:
        sde: The stochastic differential equation providing drift and diffusion terms
        t: Current time point for evaluating the SDE
        x: Current state of the system
        dt: Time step size for numerical integration
        constant_of_motion_fn: Function that computes the constant of motion and its gradients
        diffusion_scale: Scaling factor for the diffusion term

    Returns:
        tuple: (Updated state x_next, Drift term)
    """
    # Calculate regular Euler-Maruyama step
    drift = sde.f(t, x) * dt
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Compute the unconstrained increment
    dx = drift + diffusion

    # Compute gradients of the constant of motion with respect to t and x
    with torch.enable_grad():
        # Enable gradients for x and t
        x_grad = x.detach().requires_grad_(True)
        t_grad = t.detach().requires_grad_(True)

        # Compute constant of motion
        c = constant_of_motion_fn(t_grad, x_grad)

        # Compute gradients
        dcdt = torch.autograd.grad(
            c.sum(), t_grad, create_graph=False, retain_graph=True
        )[0]
        dcdx = torch.autograd.grad(
            c.sum(), x_grad, create_graph=False, retain_graph=False
        )[0]

    # Scale time derivative by dt
    dcdt = -dt * dcdt

    # Compute inner product between gradient and increment
    inner_prod = torch.sum(dcdx * dx, dim=-1, keepdim=True)

    # Compute squared norm of the gradient
    dcdx_norm_squared = (
        torch.sum(dcdx**2, dim=-1, keepdim=True) + 1e-8
    )  # Add small epsilon for stability

    # Project the increment to maintain the constant of motion
    correction_term = (inner_prod + dcdt.view(-1, 1)) * (dcdx / dcdx_norm_squared)
    dx_projected = dx - correction_term

    # Update state with projected increment
    x_next = x + dx_projected

    return x_next, drift


def integrate_constrained_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    constant_of_motion_fn,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
    temperature=1.0,
):
    """Numerically integrate an SDE while preserving a constant of motion.

    This function implements a constrained version of the Euler-Maruyama scheme,
    which projects each integration step to maintain a specified constant of motion.

    Args:
        sde: The stochastic differential equation to integrate
        x0: Initial samples to start the integration from
        num_integration_steps: Number of discretization steps for numerical integration
        energy_function: Energy function defining the target distribution
        constant_of_motion_fn: Function that computes the constant of motion and its gradients
        reverse_time: If True, integrate from t=1 to t=0 (noise→data); if False, t=0 to t=1 (data→noise)
        diffusion_scale: Scaling factor for the noise term
        no_grad: If True, disable gradient computation during integration
        time_range: Total time range for integration
        negative_time: If True, continue integration into negative time after reaching t=0
        num_negative_time_steps: Number of steps for negative time integration
        clipper: Optional clipper to stabilize score/gradient estimates

    Returns:
        torch.Tensor: Trajectory of samples during integration [num_steps, batch_size, dimensions]
    """
    # Set up the time discretization based on direction of integration
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    # Create evenly spaced time points for integration
    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x0.device
    )[
        :-1
    ]  # Remove the last point as it's handled separately

    # Initialize the current state with the input samples
    x = x0
    samples = []

    # Record the initial values of the constant of motion for monitoring
    initial_constants = constant_of_motion_fn(
        torch.tensor(start_time, device=x0.device), x0
    )
    constants = [initial_constants]

    # Conditionally disable gradient computation to save memory during inference
    with conditional_no_grad(no_grad):
        # Main integration loop: step through time points
        for i, t in enumerate(times):
            # Perform a constrained Euler-Maruyama step that preserves the constant of motion
            x, f = constrained_euler_maruyama_step(
                sde,
                t,
                x,
                time_range / num_integration_steps,
                constant_of_motion_fn,
                diffusion_scale,
            )

            # For molecular systems, enforce center of mass conservation
            if energy_function.is_molecule:
                x = remove_mean(
                    x, energy_function.n_particles, energy_function.n_spatial_dim
                )

            # Save the current state in the trajectory
            samples.append(x)

            # Monitor the constant of motion
            current_const = constant_of_motion_fn(
                torch.tensor(
                    float(t) - time_range / num_integration_steps, device=x.device
                ),
                x,
            )
            constants.append(current_const)

    # Combine all states into a single tensor representing the full trajectory
    samples = torch.stack(samples)
    constants = torch.stack(constants)

    # Optional: Continue integration into negative time (past t=0)
    if negative_time:
        print("doing negative time descent with constraint...")
        # Would need to implement a constrained version of negative_time_descent
        # For now, use the regular version
        samples_langevin = negative_time_descent(
            x,
            energy_function,
            num_steps=num_negative_time_steps,
            clipper=clipper,
            temperature=temperature,
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples, constants
