import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal
import torch
import os


class DoubleWellEnergy:
    def __init__(
        self,
        dimensionality=2,
        a=-0.5,
        b=-6.0,
        c=1.0,
        shift=0.5,  # shift the minima/transition state in the first dimension (x direction)
        device="cpu",
        *args,
        **kwargs,
    ):
        assert dimensionality == 2  # We only use the 2D version
        self._dimensionality = dimensionality
        self.device = device

        # original parameters
        self._a = a
        self._b = b
        self._c = c
        self.shift = shift
        if self._b == -6 and self._c == 1.0:
            self.means = torch.tensor([-1.7, 1.7], device=self.device)
            self.scales = torch.tensor([0.5, 0.5], device=self.device)
        else:
            raise NotImplementedError

        # ground truth minima and transition state for double well energy
        self.minima_points = torch.tensor(
            [
                [-1.7 - 0.5, 0.0],  # Left minimum
                [1.7 - 0.5, 0.0],  # Right minimum
            ]
        )
        self.transition_points = torch.tensor([[0.0 - 0.5, 0.0]])  # Transition state

    def move_to_device(self, device):
        self.component_mix = self.component_mix.to(device)
        self.means = self.means.to(device)
        self.scales = self.scales.to(device)

    def forces(self, samples):
        return -torch.vmap(torch.func.grad(self._energy))(samples)

    def _energy_dim_1(self, x_1):
        """Double well energy in x-direction: -0.5x -6x**2 + x**4"""
        # compare to Luca's example:
        # dim1 = 0.25 * (x[0]**2 - 1) ** 2 = 0.25 * x[0]**4 - 0.5 * x[0]**2 + 0.25
        # dim2 = 3 * x[1]**2
        x_1 = x_1 + self.shift
        return self._a * x_1 + self._b * x_1.pow(2) + self._c * x_1.pow(4)

    def _energy_dim_2(self, x_2):
        """Simple harmonic well energy in y-direction: 0.5y**2"""
        return 0.5 * x_2.pow(2)

    def _energy(self, x):
        """Compute energy of double well distribution.

        Args:
            x (torch.Tensor): Input samples of shape (n_samples, 2)

        Returns:
            torch.Tensor: Energy of shape (n_samples,)
        """
        if x.ndim == 1:
            e1 = self._energy_dim_1(x[0])
            e2 = self._energy_dim_2(x[1])
        else:
            e1 = self._energy_dim_1(x[:, 0])
            e2 = self._energy_dim_2(x[:, 1])
        return e1 + e2


class FourWellsEnergy:
    def __init__(
        self,
        dimensionality=2,
        a=-0.5,
        b=-6.0,
        c=1.0,
        shift=0.5,  # shift the well centers
        device="cpu",
        *args,
        **kwargs,
    ):
        assert dimensionality == 2  # We only use the 2D version
        self._dimensionality = dimensionality
        self.device = device

        # original parameters
        self._a = a
        self._b = b
        self._c = c
        self.shift = shift
        if self._b == -6 and self._c == 1.0:
            self.means = torch.tensor([-1.7, 1.7], device=self.device)
            self.scales = torch.tensor([0.5, 0.5], device=self.device)
        else:
            raise NotImplementedError

        # ground truth minima and transition state for four wells energy
        self.minima_points = torch.tensor(
            [
                [-1.7 - 0.5, 1.7],  # Left minimum
                [1.7 - 0.5, 1.7],  # Right minimum
                [-1.7 - 0.5, -1.7],  # Bottom left minimum
                [1.7 - 0.5, -1.7],  # Bottom right minimum
            ]
        )
        self.transition_points = torch.tensor(
            [
                [-1.7 - 0.5, 0],  # Left transition state
                [1.7 - 0.5, 0],  # Right transition state
                [-0.5, 1.7],  # Top transition state
                [-0.5, -1.7],  # Bottom transition state
            ]
        )

    def move_to_device(self, device):
        self.means = self.means.to(device)
        self.scales = self.scales.to(device)

    def log_prob(self, samples, temperature=None, return_aux_output=False):
        assert (
            samples.shape[-1] == self._dimensionality
        ), "`x` does not match `dimensionality`"
        log_prob = torch.squeeze(-self._energy(samples))
        if temperature is None:
            temperature = self.temperature
        log_prob = log_prob / temperature
        if return_aux_output:
            return log_prob, {}
        return log_prob

    def forces(self, samples, temperature=None):
        return -torch.vmap(torch.func.grad(self._energy))(samples)

    def _energy_dim_1(self, x_1):
        """Double well energy in x-direction: -0.5x -6x**2 + x**4"""
        x_1 = x_1 + self.shift
        return self._a * x_1 + self._b * x_1.pow(2) + self._c * x_1.pow(4)

    def _energy_dim_2(self, x_2):
        return self._a * x_2 + self._b * x_2.pow(2) + self._c * x_2.pow(4)

    def _energy(self, x):
        """Compute energy of double well distribution.

        Args:
            x (torch.Tensor): Input samples of shape (n_samples, 2)

        Returns:
            torch.Tensor: Energy of shape (n_samples,)
        """
        if x.ndim == 1:
            e1 = self._energy_dim_1(x[0])
            e2 = self._energy_dim_2(x[1])
        else:
            e1 = self._energy_dim_1(x[:, 0])
            e2 = self._energy_dim_2(x[:, 1])
        return e1 + e2

    def get_minima(self):
        """Return the locations of the four minima (well centers)."""
        return torch.tensor(
            [
                [-1.7 - self.shift, 1.7],
                [1.7 - self.shift, 1.7],
                [-1.7 - self.shift, -1.7],
                [1.7 - self.shift, -1.7],
            ],
            device=self.device,
        )

    def get_true_transition_states(self):
        """Return the saddle points between the wells."""
        return torch.tensor(
            [
                [-1.7 - self.shift, 0],
                [1.7 - self.shift, 0],
                [-self.shift, 1.7],
                [-self.shift, -1.7],
            ],
            device=self.device,
        )


# Define a function to test the energy and forces at key points
def test_energy_and_forces(energy_fn):

    # Define key points to test: minima and transition state
    # The minima are at approximately (-1.7, 0) and (1.7, 0)
    # The transition state is at approximately (0.5, 0) due to the shift
    test_points = torch.cat(
        [energy_fn.minima_points, energy_fn.transition_points], dim=0
    )

    # Calculate energies
    energies = torch.vmap(energy_fn._energy)(test_points)

    # Calculate forces
    forces = energy_fn.forces(test_points)

    # Print results
    point_names = ["Left minimum", "Right minimum", "Transition state"]
    print("Testing DoubleWellEnergy at key points:")
    print("-" * 50)
    for i, name in enumerate(point_names):
        print(f"{name} at position {test_points[i].tolist()}:")
        print(f"  Energy: {energies[i].item():.6f}")
        print(f"  Forces: {forces[i].tolist()}")
        print("-" * 50)

    # Visualize the energy landscape
    plot_energy_landscape(energy_fn)

    return energies, forces


def plot_energy_landscape(energy_fn, resolution=100, bounds=(-3, 3)):
    """Plot the 2D energy landscape with minima and transition state marked."""
    x = torch.linspace(bounds[0], bounds[1], resolution)
    y = torch.linspace(bounds[0], bounds[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Create grid of points
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate energy at each point
    energies = torch.vmap(energy_fn._energy)(points)
    Z = energies.reshape(resolution, resolution)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot contour
    contour = plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), 50, cmap="viridis")
    plt.colorbar(contour, label="Energy")

    # Mark minima and transition state
    plt.scatter(
        energy_fn.minima_points[:, 0],
        energy_fn.minima_points[:, 1],
        color="red",
        s=100,
        marker="o",
        label="Minima",
    )
    plt.scatter(
        energy_fn.transition_points[:, 0],
        energy_fn.transition_points[:, 1],
        color="white",
        s=100,
        marker="x",
        label="Transition State",
    )

    # Add force field arrows (downsampled for clarity)
    arrow_density = 15
    x_arrows = torch.linspace(bounds[0], bounds[1], arrow_density)
    y_arrows = torch.linspace(bounds[0], bounds[1], arrow_density)
    X_arrows, Y_arrows = torch.meshgrid(x_arrows, y_arrows, indexing="ij")
    points_arrows = torch.stack([X_arrows.flatten(), Y_arrows.flatten()], dim=1)

    forces_arrows = energy_fn.forces(points_arrows)
    # Normalize forces for better visualization
    forces_norm = torch.norm(forces_arrows, dim=1, keepdim=True)
    max_norm = torch.max(forces_norm)
    forces_normalized = forces_arrows / max_norm * 0.2

    plt.quiver(
        points_arrows[:, 0].numpy(),
        points_arrows[:, 1].numpy(),
        forces_normalized[:, 0].numpy(),
        forces_normalized[:, 1].numpy(),
        color="white",
        alpha=0.6,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Double Well Energy Landscape with Force Field")
    plt.legend()
    plt.tight_layout()
    figname = f"plots/baselines/{energy_fn.__class__.__name__}2d_energy_landscape.png"
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    print(f"Saved energy landscape to {figname}")


# Implement 2D well-tempered metadynamics
def run_metadynamics_2d(energy_fn):

    # Simulation parameters
    dt = 0.005  # time step
    n_steps = int(1e4)  # number of simulation steps
    mass = 1.0  # mass of the particle
    gamma = 0.5  # friction coefficient for Langevin dynamics
    temperature = 1.0  # temperature for stochastic noise

    # Well-tempered metadynamics parameters
    initial_hill_height = 1.0  # initial hill height (w0)
    hill_width = 0.3  # width (sigma) of each Gaussian hill
    deposit_interval = 10  # deposit a hill every deposit_interval steps
    delta_T = 0.5  # effective bias "temperature" that controls tempering

    # Initialize arrays to store trajectory and times
    trajectory = torch.zeros((n_steps, 2))
    times = torch.zeros(n_steps)

    # Initial conditions: start at one of the minima
    x = torch.tensor([-1.7, 0.0])  # Start at the left minimum
    v = torch.zeros(2)

    # Lists to store the hill positions and heights (bias potential)
    hill_positions = []
    hill_heights = []

    # Store energies to identify transition states
    energies = torch.zeros(n_steps)

    # For reproducibility
    torch.manual_seed(42)

    # Define bias potential function
    def bias_potential(pos, hill_positions, hill_heights, hill_width):
        """
        Compute the total bias potential at position pos due to deposited hills.
        Each hill is a Gaussian of the form:
           w * exp(-||pos - pos_i||^2 / (2 * sigma^2))
        """
        if not hill_positions:
            return torch.tensor(0.0)

        hills_tensor = torch.stack(hill_positions)
        heights_tensor = torch.tensor(hill_heights)

        # Calculate squared distances
        diff = pos.unsqueeze(0) - hills_tensor
        sq_distances = torch.sum(diff**2, dim=1)

        # Calculate Gaussian contributions
        gaussians = torch.exp(-sq_distances / (2 * hill_width**2))

        # Sum up all contributions
        bias = torch.sum(heights_tensor * gaussians)
        return bias

    # Define bias force function
    def bias_force(pos, hill_positions, hill_heights, hill_width):
        """
        Compute the force due to the bias potential.
        F_bias(pos) = -∇V_bias(pos)
        """
        if not hill_positions:
            return torch.zeros(2)

        hills_tensor = torch.stack(hill_positions)
        heights_tensor = torch.tensor(hill_heights)

        # Calculate differences and squared distances
        diff = pos.unsqueeze(0) - hills_tensor
        sq_distances = torch.sum(diff**2, dim=1)

        # Calculate Gaussian contributions
        gaussians = torch.exp(-sq_distances / (2 * hill_width**2))

        # Calculate force components
        prefactor = heights_tensor / hill_width**2
        force_components = prefactor.unsqueeze(1) * diff * gaussians.unsqueeze(1)

        # Sum up all contributions
        force = torch.sum(force_components, dim=0)
        return force

    # Main simulation loop using Langevin dynamics
    for step in tqdm(range(n_steps)):
        t = step * dt
        times[step] = t

        # Total force: from the double well and the bias
        F_system = energy_fn.forces(x.unsqueeze(0)).squeeze(0)
        F_bias = bias_force(x, hill_positions, hill_heights, hill_width)
        F_total = F_system + F_bias

        # Include Langevin thermostat: damping and stochastic noise
        noise = torch.sqrt(torch.tensor(2 * gamma * temperature / dt)) * torch.randn(2)

        # Update velocity and position (simple Euler integration)
        v = v + dt * (F_total - gamma * v + noise) / mass
        x = x + dt * v

        trajectory[step] = x

        # Store the original energy (without bias) at this position
        energies[step] = energy_fn._energy(x)

        # Deposit a hill every deposit_interval steps using well-tempered scaling
        if step % deposit_interval == 0:
            # Evaluate current bias potential at x
            current_bias = bias_potential(x, hill_positions, hill_heights, hill_width)

            # Scale hill height using well-tempered protocol
            effective_hill_height = initial_hill_height * torch.exp(
                -current_bias / delta_T
            )

            hill_positions.append(x.clone())
            hill_heights.append(effective_hill_height.item())

    # Find transition states by identifying energy maxima between basins
    # First, smooth the energy profile to reduce noise
    window_size = 50
    smoothed_energies = torch.zeros_like(energies)
    for i in range(len(energies)):
        start = max(0, i - window_size // 2)
        end = min(len(energies), i + window_size // 2)
        smoothed_energies[i] = torch.mean(energies[start:end])

    # Find local maxima in the energy profile
    smoothed_energies_np = smoothed_energies.numpy()
    peaks, _ = scipy.signal.find_peaks(
        smoothed_energies_np,
        #  prominence=0.5, # Adjust prominence as needed
    )
    print(f"Found {len(peaks)} transition states")

    # Extract transition state positions from the trajectory
    transition_states = trajectory[peaks]

    ####################################################################################################################
    # Plot the trajectory and free energy landscape
    plt.figure(figsize=(18, 6))

    # Plot the trajectory
    plt.subplot(1, 3, 1)
    plt.plot(
        trajectory[:, 0].numpy(),
        trajectory[:, 1].numpy(),
        "b-",
        alpha=0.6,
        linewidth=0.8,
    )
    plt.scatter(
        trajectory[0, 0].numpy(),
        trajectory[0, 1].numpy(),
        color="green",
        s=100,
        label="Start",
    )
    plt.scatter(
        trajectory[-1, 0].numpy(),
        trajectory[-1, 1].numpy(),
        color="red",
        s=100,
        label="End",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Metadynamics Trajectory")
    plt.legend()

    # Plot the original energy landscape
    plt.subplot(1, 3, 2)

    # Create grid for visualization
    resolution = 100
    bounds = [-3, 3]
    x_vals = torch.linspace(bounds[0], bounds[1], resolution)
    y_vals = torch.linspace(bounds[0], bounds[1], resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate original energy
    original_energies = torch.vmap(energy_fn._energy)(grid_points)
    original_energy_grid = original_energies.reshape(resolution, resolution)

    # Plot original energy landscape
    contour_orig = plt.contourf(
        X.numpy(), Y.numpy(), original_energy_grid.numpy(), 50, cmap="viridis"
    )
    plt.colorbar(contour_orig, label="Energy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original Energy Landscape")

    if peaks is not None and len(peaks) > 0:
        plt.scatter(
            transition_states[:, 0].numpy(),
            transition_states[:, 1].numpy(),
            color="white",
            s=100,
            marker="x",
            label="Transition State",
        )

    # Plot the biased energy landscape
    plt.subplot(1, 3, 3)

    # Calculate bias potential for each grid point
    bias_energies = torch.zeros(grid_points.shape[0])
    for i, point in enumerate(grid_points):
        bias_energies[i] = bias_potential(
            point, hill_positions, hill_heights, hill_width
        )

    bias_energy_grid = bias_energies.reshape(resolution, resolution)
    total_energy_grid = original_energy_grid + bias_energy_grid

    # Plot biased energy landscape
    contour_biased = plt.contourf(
        X.numpy(), Y.numpy(), total_energy_grid.numpy(), 50, cmap="viridis"
    )
    plt.colorbar(contour_biased, label="Energy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Biased Energy Landscape")

    plt.tight_layout()
    figname = f"plots/baselines/metadynamics_welltempered_{energy_fn.__class__.__name__}2d_energy_landscape.png"
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    print(f"Saved metadynamics results to {figname}")

    ####################################################################################################################
    # plot the energy trajectory
    plt.close()
    # Plot the energy trajectory over time
    plt.figure(figsize=(10, 6))

    # Plot the energy values over time
    plt.plot(times.numpy(), energies.numpy(), "gray", linewidth=1.5)

    # Mark transition states if any were detected
    if peaks is not None and len(peaks) > 0:
        transition_times = [times[idx].item() for idx in peaks]
        transition_energies = [energies[idx].item() for idx in peaks]
        plt.scatter(
            transition_times,
            transition_energies,
            color="red",
            s=80,
            marker="o",
            label="Found Transition States",
        )

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Trajectory Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    figname = f"plots/baselines/metadynamics_welltempered_{energy_fn.__class__.__name__}2d_energy_trajectory.png"
    plt.savefig(figname)
    print(f"Saved energy trajectory to {figname}")

    return trajectory, hill_positions, hill_heights, transition_states


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_energy_and_forces(DoubleWellEnergy())
    test_energy_and_forces(FourWellsEnergy())

    run_metadynamics_2d(DoubleWellEnergy())
    run_metadynamics_2d(FourWellsEnergy())
