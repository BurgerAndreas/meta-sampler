import numpy as np
import torch
from typing import Callable
from dem.energies.base_energy_function import BaseEnergyFunction
import matplotlib.pyplot as plt
import itertools

# https://github.com/lollcat/fab-torch/blob/master/fab/target_distributions/double_well.py


def rejection_sampling(
    n_samples: int,
    proposal: torch.distributions.Distribution,
    target_log_prob_fn: Callable,
    k: float,
) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample((n_samples * 10,))
    u_0 = (
        torch.distributions.Uniform(0, k * torch.exp(proposal.log_prob(z_0)))
        .sample()
        .to(z_0)
    )
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(
            required_samples, proposal, target_log_prob_fn, k
        )
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


class DoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        a=-0.5,
        b=-6.0,
        c=1.0,
        device="cpu",
        is_molecule=False,
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=1,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        assert dimensionality == 2  # We only use the 2D version
        self._dimensionality = dimensionality
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        self.data_path_train = data_path_train
        self.device = device

        # original parameters
        self._a = a
        self._b = b
        self._c = c
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define proposal params
            self.component_mix = torch.tensor([0.2, 0.8], device=device)
            self.means = torch.tensor([-1.7, 1.7], device=device)
            self.scales = torch.tensor([0.5, 0.5], device=device)

        # this will setup test, train, val sets
        super().__init__(
            dimensionality=dimensionality,
            plotting_bounds=(-4, 4),
            plotting_buffer_sample_size=plotting_buffer_sample_size,
            plot_samples_epoch_period=plot_samples_epoch_period,
            should_unnormalize=should_unnormalize,
            train_set_size=train_set_size,
            test_set_size=test_set_size,
            val_set_size=val_set_size,
            data_path_train=data_path_train,
        )

    def move_to_device(self, device):
        self.component_mix = self.component_mix.to(device)
        self.means = self.means.to(device)
        self.scales = self.scales.to(device)

    def __call__(self, samples, temperature=None, return_aux_output=False):
        return self.log_prob(
            samples, temperature=temperature, return_aux_output=return_aux_output
        )

    def log_prob(self, samples, temperature=None, return_aux_output=False):
        log_prob = torch.squeeze(-self.energy(samples, temperature=temperature))
        if return_aux_output:
            return log_prob, {}
        return log_prob

    def energy(self, samples, temperature=None):
        assert (
            samples.shape[-1] == self._dimensionality
        ), "`x` does not match `dimensionality`"
        if temperature is None:
            temperature = 1.0
        return self._energy(samples) / temperature

    def force(self, samples, temperature=None):
        samples = samples.requires_grad_(True)
        e = self.energy(samples, temperature=temperature)
        return -torch.autograd.grad(e.sum(), samples)[0]

    def _energy_dim_1(self, x_1):
        """-0.5x -6x**2 + x**4"""
        # compare:
        # dim1 = 0.25 * (x[0]**2 - 1) ** 2 = 0.25 * x[0]**4 - 0.5 * x[0]**2 + 0.25
        # dim2 = 3 * x[1]**2
        return self._a * x_1 + self._b * x_1.pow(2) + self._c * x_1.pow(4)

    def _energy_dim_2(self, x_2):
        """0.5x**2"""
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

    def sample_first_dimension(self, shape):
        assert len(shape) == 1
        # see fab.sampling_methods.rejection_sampling_test.py
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            # Define target.
            def target_log_prob(x):
                return -(x**4) + 6 * x**2 + 1 / 2 * x

            TARGET_Z = 11784.50927

            # Define proposal
            mix = torch.distributions.Categorical(self.component_mix)
            com = torch.distributions.Normal(self.means, self.scales)

            proposal = torch.distributions.MixtureSameFamily(
                mixture_distribution=mix, component_distribution=com
            )

            k = TARGET_Z * 3

            samples = rejection_sampling(shape[0], proposal, target_log_prob, k)
            return samples
        else:
            raise NotImplementedError

    def sample(self, shape):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            dim1_samples = self.sample_first_dimension(shape)
            dim2_samples = torch.distributions.Normal(
                torch.tensor(0.0).to(dim1_samples.device),
                torch.tensor(1.0).to(dim1_samples.device),
            ).sample(shape)
            return torch.stack([dim1_samples, dim2_samples], dim=-1)
        else:
            raise NotImplementedError

    @property
    def log_Z_2D(self):
        if self._a == -0.5 and self._b == -6 and self._c == 1.0:
            log_Z_dim0 = np.log(11784.50927)
            log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
            return log_Z_dim0 + log_Z_dim1
        else:
            raise NotImplementedError

    #####################################################################
    # added for DEM
    #####################################################################

    def get_minima(self):
        return torch.tensor([[-1.7, 0.0], [1.7, 0.0]], device=self.device)

    def get_true_transition_states(self):
        return torch.tensor([[0.0, 0.0]], device=self.device)

    def setup_test_set(self):
        """Sets up test dataset by sampling from GMM.
        Used in train.py and eval.py during model testing.

        Returns:
            torch.Tensor: Test dataset tensor
        """
        test_sample = self.sample((self.test_set_size,))
        return test_sample

    def setup_train_set(self):
        """Sets up training dataset by sampling from GMM or loading from file.
        Used in train.py during model training.

        Returns:
            torch.Tensor: Training dataset tensor
        """
        if self.data_path_train is None:
            train_samples = self.normalize(self.sample((self.train_set_size,)))

        else:
            # Assume the samples we are loading from disk are already normalized.
            # This breaks if they are not.

            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)

            data = torch.tensor(data, device=self.device)

        return train_samples

    def setup_val_set(self):
        """Sets up validation dataset by sampling from GMM.
        Used in train.py during model validation.

        Returns:
            torch.Tensor: Validation dataset tensor
        """
        val_samples = self.sample((self.val_set_size,))
        return val_samples

    def get_hessian_eigenvalues_on_grid(
        self, grid_width_n_points=200, plotting_bounds=None
    ):
        """Compute eigenvalues of the Hessian on a grid.

        Args:
            grid_width_n_points (int): Number of points in each dimension for grid
            plotting_bounds (tuple, optional): Plot bounds as (min, max) tuple

        Returns:
            tuple: Grid points and eigenvalues tensors
        """
        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds

        x_points_dim1 = torch.linspace(
            plotting_bounds[0], plotting_bounds[1], grid_width_n_points
        )
        x_points_dim2 = torch.linspace(
            plotting_bounds[0], plotting_bounds[1], grid_width_n_points
        )
        x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))

        # Move to device
        x_points = x_points.to(self.device)

        # Compute eigenvalues using torch.func
        def get_energy(x):
            return self._energy(x)

        # Use functorch to compute Hessian directly
        if len(x_points.shape) == 1:
            # Handle single sample
            hessian = torch.func.hessian(get_energy)(x_points)
        else:
            # Handle batched inputs using vmap
            hessian = torch.vmap(torch.func.hessian(get_energy))(x_points)

        # Compute eigenvalues
        batched_eigenvalues, _ = torch.linalg.eigh(hessian)

        # Sort eigenvalues
        sorted_indices = torch.argsort(batched_eigenvalues, dim=-1)
        batched_eigenvalues = torch.gather(batched_eigenvalues, -1, sorted_indices)

        # Reshape for plotting
        eigenvalues_grid = batched_eigenvalues.reshape(
            grid_width_n_points, grid_width_n_points, self._dimensionality
        )
        x_grid = x_points_dim1.reshape(-1, 1).repeat(1, grid_width_n_points)
        y_grid = x_points_dim2.reshape(1, -1).repeat(grid_width_n_points, 1)

        return x_grid, y_grid, eigenvalues_grid

    def plot_hessian_eigenvalues(self, grid_width_n_points=200, plotting_bounds=None, name=None):
        """Plot the first two eigenvalues of the Hessian on a grid.

        Args:
            grid_width_n_points (int): Number of points in each dimension for grid
            plotting_bounds (tuple, optional): Plot bounds as (min, max) tuple

        Returns:
            matplotlib.figure.Figure: Figure with eigenvalue plots
        """
        # Move to CPU for plotting
        device_backup = self.device
        self.move_to_device("cpu")

        x_grid, y_grid, eigenvalues_grid = self.get_hessian_eigenvalues_on_grid(
            grid_width_n_points=grid_width_n_points, plotting_bounds=plotting_bounds
        )
        eigenvalues_grid = eigenvalues_grid.cpu().numpy()

        # Create figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        title = "Hessian eigenvalues"
        if name is not None:
            title = f"{title}: {name}"
        fig.suptitle(title)

        # Plot first eigenvalue
        im1 = axs[0].pcolormesh(
            x_grid, y_grid, eigenvalues_grid[:, :, 0], cmap="viridis", shading="auto"
        )
        axs[0].set_title("First (smallest) eigenvalue")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(im1, ax=axs[0])

        # Plot second eigenvalue
        im2 = axs[1].pcolormesh(
            x_grid, y_grid, eigenvalues_grid[:, :, 1], cmap="viridis", shading="auto"
        )
        axs[1].set_title("Second eigenvalue")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()

        # Restore device
        self.move_to_device(device_backup)

        return fig
    
    def plot_energy_crossection(self, x_range=(-4, 4), n_points=200, y_value=0.0, name=None):
        """Plot a horizontal cross-section of the energy landscape at a specified y-value.
        
        Args:
            x_range (tuple): Range of x values to plot (min, max)
            n_points (int): Number of points to sample along the x-axis
            y_value (float): The y-value at which to take the cross-section
            
        Returns:
            matplotlib.figure.Figure: Figure with the energy cross-section plot
        """
        self.move_to_device("cpu")
        
        # Create x points for the cross-section
        x_points = torch.linspace(x_range[0], x_range[1], n_points)
        
        # Create samples with fixed y value
        samples = torch.zeros((n_points, 2))
        samples[:, 0] = x_points
        samples[:, 1] = y_value
        
        # Compute energy values
        energy_values = self.energy(samples).detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot energy cross-section
        ax.plot(x_points, energy_values)
        title = f"Energy cross-section at y = {y_value}"
        if name is not None:
            title = f"{title}: {name}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("Energy")
        ax.grid(True)
        
        # Mark minima and saddle points if they're within the cross-section
        minima = self.get_minima()
        saddle_points = self.get_true_transition_states()
        
        # Filter points that are on or very close to our cross-section
        epsilon = 1e-6
        for point in minima:
            if abs(point[1].item() - y_value) < epsilon:
                ax.plot(point[0].item(), self.energy(point).item(), 'ro', markersize=8, label='Minimum')
                
        for point in saddle_points:
            if abs(point[1].item() - y_value) < epsilon:
                ax.plot(point[0].item(), self.energy(point).item(), 'go', markersize=8, label='Saddle point')
        
        # Add legend if we plotted any special points
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
            
        plt.tight_layout()

        # Restore device
        self.move_to_device(self.device)

        return fig


if __name__ == "__main__":
    # Test that rejection sampling is work as desired.
    import matplotlib.pyplot as plt

    target = DoubleWellEnergy(2)

    x_linspace = torch.linspace(-4, 4, 200)

    Z_dim_1 = 11784.50927
    samples = target.sample((10000,))
    p_1 = torch.exp(-target._energy_dim_1(x_linspace))
    # plot first dimension vs normalised log prob
    plt.plot(x_linspace, p_1 / Z_dim_1, label="p_1 normalised")
    plt.hist(samples[:, 0], density=True, bins=100, label="sample density")
    plt.legend()
    plt.show()

    # Now dimension 2.
    Z_dim_2 = (2 * torch.pi) ** 0.5
    p_2 = torch.exp(-target._energy_dim_2(x_linspace))
    # plot first dimension vs normalised log prob
    plt.plot(x_linspace, p_2 / Z_dim_2, label="p_2 normalised")
    plt.hist(samples[:, 1], density=True, bins=100, label="sample density")
    plt.legend()
    plt.show()
