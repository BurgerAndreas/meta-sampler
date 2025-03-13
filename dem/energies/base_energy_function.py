from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot after setting backend

plt.ioff()  # Turn off interactive model

from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image
from dem.utils.plotting import plot_fn, plot_marginal_pair
import traceback
import copy
from tqdm import tqdm
import itertools

class BaseEnergyFunction(ABC):
    """Base class for energy functions used in DEM.

    This class provides the basic interface and functionality for energy functions,
    including data normalization, sampling from datasets, and logging.

    Args:
        dimensionality (int): Dimension of the input space
        is_molecule (bool, optional): Whether this energy function is for a molecule. Defaults to False.
        normalization_min (float, optional): Minimum value for normalization. Defaults to None.
        normalization_max (float, optional): Maximum value for normalization. Defaults to None.
        
    When implementing a new energy function, you need to implement the following methods:
    - log_prob
    - _energy
    And consider implementing the following methods:
    - move_to_device
    """

    def __init__(
        self,
        dimensionality: int = 2,
        is_molecule: Optional[bool] = False,
        normalization_min: Optional[float] = None,
        normalization_max: Optional[float] = None,
        plotting_bounds: Optional[tuple] = (-1.4 * 40, 1.4 * 40),
        plotting_buffer_sample_size: int = 512,
        plot_samples_epoch_period: int = 5,
        should_unnormalize: bool = False,
        train_set_size: int = 100000,
        test_set_size: int = 2000,
        val_set_size: int = 2000,
        data_path_train: Optional[str] = None,
        temperature: float = 1.0,
    ):
        self._dimensionality = dimensionality
        self._plotting_bounds = plotting_bounds

        self.normalization_min = normalization_min
        self.normalization_max = normalization_max

        self._is_molecule = is_molecule
        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period
        self.should_unnormalize = should_unnormalize

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        self.data_path_train = data_path_train

        self.temperature = temperature

        self._test_set = self.setup_test_set()
        self._val_set = self.setup_val_set()
        self._train_set = None
        
        self.name = self.__class__.__name__

    def setup_test_set(self) -> Optional[torch.Tensor]:
        """Sets up the test dataset.

        Returns:
            Optional[torch.Tensor]: Test dataset tensor or None
        """
        return None

    def setup_train_set(self) -> Optional[torch.Tensor]:
        """Sets up the training dataset.

        Returns:
            Optional[torch.Tensor]: Training dataset tensor or None
        """
        return None

    def setup_val_set(self) -> Optional[torch.Tensor]:
        """Sets up the validation dataset.

        Returns:
            Optional[torch.Tensor]: Validation dataset tensor or None
        """
        return None

    @property
    def _can_normalize(self) -> bool:
        """Whether normalization can be performed.

        Returns:
            bool: True if normalization bounds are set
        """
        return self.normalization_min is not None and self.normalization_max is not None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes input tensor to [-1, 1] range.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Normalized tensor
        """
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        # [ 0, 1 ]
        x = (x - mins) / (maxs - mins + 1e-5)
        # [ -1, 1 ]
        return x * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalizes input tensor from [-1, 1] range.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Unnormalized tensor
        """
        if x is None or not self._can_normalize:
            return x

        mins = self.normalization_min
        maxs = self.normalization_max

        x = (x + 1) / 2
        return x * (maxs - mins) + mins

    def move_to_device(self, device):
        pass

    def get_dataset_fig(
        self,
        samples,
        gen_samples=None,
        n_contour_levels=50,
        plotting_bounds=None,
        plot_minima=False,
        plot_style="contours",
        plot_prob_kwargs={},
        plot_sample_kwargs={},
        colorbar=False,
        quantity="log_prob",
    ):
        """Creates side-by-side visualization of buffer and generated samples.
        Used in train.py for comparing sample distributions.

        Args:
            samples (torch.Tensor): Buffer samples to plot
            gen_samples (torch.Tensor, optional): Generated samples to plot. Defaults to None
            plotting_bounds (tuple, optional): Plot bounds. Defaults to (-1.4*40, 1.4*40)

        Returns:
            numpy.ndarray: Plot as image array
        """
        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.move_to_device("cpu")
        plot_fn(
            self.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=200,
            plot_kwargs=plot_prob_kwargs,
            colorbar=colorbar,
            quantity=quantity,
        )

        # plot dataset samples
        if samples is not None:
            plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
            axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_fn(
                self.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                plot_style=plot_style,
                n_contour_levels=50,
                grid_width_n_points=200,
                plot_kwargs=plot_prob_kwargs,
                colorbar=colorbar,
                quantity=quantity,
            )
            # plot generated samples
            plot_marginal_pair(
                gen_samples,
                ax=axs[1],
                bounds=plotting_bounds,
                plot_kwargs=plot_sample_kwargs,
            )
            axs[1].set_title("Generated samples")

        if plot_minima:
            if hasattr(self, "get_minima"):
                minima = self.get_minima()
                axs[1].scatter(*minima.detach().cpu().T, color="red", marker="x")
            # axs[1].legend()

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.move_to_device(self.device)

        return fig_to_image(fig)

    def get_single_dataset_fig(
        self,
        samples,
        name=None,
        n_contour_levels=50,
        plotting_bounds=None,
        plot_minima=False,
        grid_width_n_points=200,
        plot_style="contours",
        with_legend=False,
        plot_prob_kwargs={},
        plot_sample_kwargs={},
        colorbar=False,
        quantity="log_prob",
        ax=None,
        return_fig=False,
    ):
        """Creates visualization of samples against GMM contours.
        Used in train.py for sample visualization.

        Args:
            samples (torch.Tensor): Samples to plot
            name (str): Title for plot
            plotting_bounds (tuple, optional): Plot bounds. Defaults to (-1.4*40, 1.4*40)

        Returns:
            numpy.ndarray: Plot as image array
        """
        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds
        if ax is None:
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.get_figure()

        self.move_to_device("cpu")
        plot_fn(
            self.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=grid_width_n_points,
            plot_kwargs=plot_prob_kwargs,
            colorbar=colorbar,
            quantity=quantity,
        )
        if samples is not None:
            plot_marginal_pair(
                samples, ax=ax, bounds=plotting_bounds, plot_kwargs=plot_sample_kwargs
            )
        if name is not None:
            ax.set_title(f"{name}")

        if plot_minima:
            minima = self.get_minima()
            ax.scatter(*minima.detach().cpu().T, color="red", marker="x")
            # ax.legend()

        if with_legend:
            ax.legend()

        self.move_to_device(self.device)

        if return_fig:
            return fig
        else:
            return fig_to_image(fig)

    def sample_test_set(
        self, num_points: int, normalize: bool = False, full: bool = False
    ) -> Optional[torch.Tensor]:
        """Samples points from the test set.

        Args:
            num_points (int): Number of points to sample
            normalize (bool, optional): Whether to normalize samples. Defaults to False.
            full (bool, optional): Whether to return full dataset. Defaults to False.

        Returns:
            Optional[torch.Tensor]: Sampled points or None if no test set
        """
        if self.test_set is None:
            return None

        if full:
            outs = self.test_set
        else:
            idxs = torch.randperm(len(self.test_set))[:num_points]
            outs = self.test_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_train_set(
        self, num_points: int, normalize: bool = False
    ) -> Optional[torch.Tensor]:
        """Samples points from the training set.

        Args:
            num_points (int): Number of points to sample
            normalize (bool, optional): Whether to normalize samples. Defaults to False.

        Returns:
            Optional[torch.Tensor]: Sampled points or None if no training set
        """
        if self.train_set is None:
            self._train_set = self.setup_train_set()

        idxs = torch.randperm(len(self.train_set))[:num_points]
        outs = self.train_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    def sample_val_set(
        self, num_points: int, normalize: bool = False
    ) -> Optional[torch.Tensor]:
        """Samples points from the validation set.

        Args:
            num_points (int): Number of points to sample
            normalize (bool, optional): Whether to normalize samples. Defaults to False.

        Returns:
            Optional[torch.Tensor]: Sampled points or None if no validation set
        """
        if self.val_set is None:
            return None

        idxs = torch.randperm(len(self.val_set))[:num_points]
        outs = self.val_set[idxs]
        if normalize:
            outs = self.normalize(outs)

        return outs

    @property
    def dimensionality(self) -> int:
        """Dimension of the input space.

        Returns:
            int: Input dimensionality
        """
        return self._dimensionality

    @property
    def is_molecule(self) -> Optional[bool]:
        """Whether this energy function is for a molecule.

        Returns:
            Optional[bool]: True if molecular system
        """
        return self._is_molecule

    @property
    def test_set(self) -> Optional[torch.Tensor]:
        """The test dataset.

        Returns:
            Optional[torch.Tensor]: Test dataset tensor
        """
        return self._test_set

    @property
    def val_set(self) -> Optional[torch.Tensor]:
        """The validation dataset.

        Returns:
            Optional[torch.Tensor]: Validation dataset tensor
        """
        return self._val_set

    @property
    def train_set(self) -> Optional[torch.Tensor]:
        """The training dataset.

        Returns:
            Optional[torch.Tensor]: Training dataset tensor
        """
        return self._train_set

    # @abstractmethod
    def __call__(
        self, samples: torch.Tensor, temperature: Optional[float] = None, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the normalized (pseudo-)energy function at given samples.

        Args:
            samples (torch.Tensor): Input points

        Returns:
            torch.Tensor: Energy values at input points
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return self.log_prob(samples, temperature=temperature, return_aux_output=return_aux_output)
    
    def log_prob(
        self, samples: torch.Tensor, temperature: Optional[float] = None, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the unnormalized log probability of the (pseudo-)energy function at given samples.

        Args:
            samples (torch.Tensor): Input points
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False

        Returns:
            torch.Tensor: Log probability values at input points
        """
        raise NotImplementedError(f"Energy function {self.__class__.__name__} must implement `log_prob()`")
    
    def _energy(
        self, samples: torch.Tensor, temperature: Optional[float] = None, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Energy of the unnormalized (physical) potential. Used in GAD to compute forces/Hessian.

        Args:
            samples (torch.Tensor): Input points
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False

        Returns:
            torch.Tensor: Energy values at input points
        """
        raise NotImplementedError(f"GAD energy function {self.__class__.__name__} must implement `_energy()`")

    def energy(self, samples: torch.Tensor, temperature: Optional[float] = None, return_aux_output: bool = False) -> torch.Tensor:
        """Energy=-logprob of unnormalized (pseudo-)potential. 
        Convinience fuction. Superflous with `log_prob()`. Used for plotting.
        """
        if return_aux_output:
            log_prob, aux_output = self.log_prob(samples, temperature=temperature, return_aux_output=True)
            return -log_prob, aux_output
        return -self.log_prob(samples, temperature=temperature)

    def score(self, samples: torch.Tensor) -> torch.Tensor:
        """Computes the score (gradient of energy) at given samples.

        Args:
            samples (torch.Tensor): Input points

        Returns:
            torch.Tensor: Score values at input points
        """
        grad_fxn = torch.func.grad(self.__call__)
        vmapped_grad = torch.vmap(grad_fxn)
        return vmapped_grad(samples)

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        replay_buffer: ReplayBuffer = None,
        wandb_logger: WandbLogger = None,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        prefix: str = "",
        return_fig: bool = False,
    ) -> None:
        """Logs metrics and visualizations at the end of each epoch.

        Args:
            latest_samples (torch.Tensor): Most recent generated samples
            latest_energies (torch.Tensor): Energy values for latest samples
            replay_buffer (ReplayBuffer): Replay buffer containing past samples
            wandb_logger (WandbLogger): Logger for metrics and visualizations
        """
        if wandb_logger is None and not return_fig:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                if latest_samples is not None:
                    latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(
                        unprioritized_buffer_samples
                    )

            if unprioritized_buffer_samples is not None:
                buffer_samples, _, _ = replay_buffer.sample(
                    self.plotting_buffer_sample_size
                )
                if self.should_unnormalize:
                    buffer_samples = self.unnormalize(buffer_samples)

                samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)

                wandb_logger.log_image(
                    f"{prefix}unprioritized_buffer_samples", [samples_fig]
                )

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(
                    unprioritized_buffer_samples, cfm_samples
                )

                wandb_logger.log_image(
                    f"{prefix}cfm_generated_samples", [cfm_samples_fig]
                )

            if latest_samples is not None:
                assert torch.isfinite(latest_samples).all(), f"Max value: {latest_samples.max()}, Min value: {latest_samples.min()}"
                print(f"epoch={self.curr_epoch}. Samples: max={latest_samples.max()}, min={latest_samples.min()}. nan={torch.sum(torch.isnan(latest_samples))}")
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)

                wandb_logger.log_image(
                    f"{prefix}generated_samples_scatter", [fig_to_image(fig)]
                )
                img = self.get_single_dataset_fig(
                    latest_samples, "dem_generated_samples"
                )
                wandb_logger.log_image(f"{prefix}generated_samples", [img])

            plt.close()

        self.curr_epoch += 1

    def save_samples(
        self,
        samples: torch.Tensor,
        dataset_name: str,
    ) -> None:
        """Saves samples to disk.

        Args:
            samples (torch.Tensor): Samples to save
            dataset_name (str): Name for the saved dataset
        """
        np.save(f"{dataset_name}_samples.npy", samples.cpu().numpy())
    
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

        # Use functorch to compute Hessian directly
        if len(x_points.shape) == 1:
            # Handle single sample
            hessian = torch.func.hessian(self._energy)(x_points)
        else:
            # Handle batched inputs using vmap
            hessian = torch.vmap(torch.func.hessian(self._energy))(x_points)

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
        if self.dimensionality != 2:
            print(f"Hessian eigenvalues are only defined for 2D systems, but this system has {self.dimensionality} dimensions")
            return

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

        return fig
    
    def plot_gradient(self, grid_width_n_points=200, grid_width_n_points_vector_field=10, plotting_bounds=None, name=None):
        """Plot the gradient of the potential on a grid.

        Args:
            grid_width_n_points (int): Number of points in each dimension for grid
            plotting_bounds (tuple, optional): Plot bounds as (min, max) tuple
            name (str, optional): Name to include in the plot title
            subsample_factor (int, optional): Factor to subsample the vector field for clarity

        Returns:
            matplotlib.figure.Figure: Figure with gradient vector field plot
        """
        if self.dimensionality != 2:
            print(f"Gradient plotting is only defined for 2D systems, but this system has {self.dimensionality} dimensions")
            return

        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds

        # Create grid for energy contour plot (higher resolution)
        contour_n_points = grid_width_n_points
        x_contour = torch.linspace(plotting_bounds[0], plotting_bounds[1], contour_n_points)
        y_contour = torch.linspace(plotting_bounds[0], plotting_bounds[1], contour_n_points)
        X_contour, Y_contour = np.meshgrid(x_contour.cpu().numpy(), y_contour.cpu().numpy())
        
        # Create points for energy evaluation
        contour_points = torch.tensor(list(itertools.product(x_contour, y_contour)))
        contour_points = contour_points.to(self.device)
        
        # Compute energy values for contour plot
        energy_values = self.energy(contour_points).reshape(contour_n_points, contour_n_points).detach().cpu().numpy()
        
        ############################################################################################################
        # Create smaller grid for vector field
        x_points_dim1 = torch.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points_vector_field)
        x_points_dim2 = torch.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points_vector_field)
        x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
        x_points = x_points.to(self.device)
        
        def scalar_log_prob(x):
            # torch.func.grad requires a function that returns a single scalar value for each individual input point
            # For each single grid point x (which is a tensor of shape [2]):
            # We add a batch dimension with x.unsqueeze(0) to make it [1, 2]
            # When we call self.log_prob(x.unsqueeze(0)), it returns a tensor of shape [1] (a batch with one value)
            # We extract that single value with [0] to get a scalar (shape [])
            return -self.log_prob(x.unsqueeze(0)).squeeze(0)
        
        grad_values_grid = torch.vmap(torch.func.grad(scalar_log_prob))(x_points)
        
        # Reshape for plotting
        grad_values_grid = grad_values_grid.reshape(
            grid_width_n_points_vector_field, grid_width_n_points_vector_field, self._dimensionality
        )
        x_grid = x_points_dim1.reshape(-1, 1).repeat(1, grid_width_n_points_vector_field)
        y_grid = x_points_dim2.reshape(1, -1).repeat(grid_width_n_points_vector_field, 1)
        grad_values_grid = grad_values_grid.cpu().numpy()

        ############################################################################################################
        # Create figure 
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot energy contours as background
        contour = ax.contourf(X_contour, Y_contour, energy_values, levels=50, cmap='viridis', alpha=0.6)
        cbar_contour = fig.colorbar(contour, ax=ax)
        cbar_contour.set_label('Energy')
        
        # Subsample the grid for clearer vector field visualization
        U = grad_values_grid[:, :, 0]  # x component
        V = grad_values_grid[:, :, 1]  # y component
        
        # Plot the vector field with light gray color and magnitude as vector length
        quiver = ax.quiver(
            x_grid, y_grid, 
            U, V, 
            scale=0.01,  # Use scale=1.0 to show actual magnitudes
            pivot='mid',
            alpha=0.7,
            width=0.004,
            color='lightgray'  # Use light gray for all vectors
        )
        
        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title = "Gradient Vector Field"
        if name is not None:
            title = f"{title}: {name}"
        ax.set_title(title)
        
        # Add minima points if available
        try:
            minima = self.get_minima()
            if minima is not None:
                ax.scatter(*minima.detach().cpu().T, color="red", marker="x", s=100, label="Minima")
                ax.legend()
        except:
            pass

        plt.tight_layout()

        return fig
    
    def plot_energy_crossection(self, n_points=200, plotting_bounds=None, y_value=0.0, name=None):
        """Plot a horizontal cross-section of the energy landscape at a specified y-value.
        
        Args:
            n_points (int): Number of points to sample along the x-axis
            y_value (float): The y-value at which to take the cross-section
            
        Returns:
            matplotlib.figure.Figure: Figure with the energy cross-section plot
        """
        if self.dimensionality != 2:
            print(f"Energy cross-section is only defined for 2D systems, but this system has {self.dimensionality} dimensions")
            return
        
        self.move_to_device("cpu")
        
        # Create x points for the cross-section
        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds
        x_points = torch.linspace(plotting_bounds[0], plotting_bounds[1], n_points)
        
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
        title = f"Energy cross-section at y={y_value:.1f}"
        if name is not None:
            title = f"{title}: {name}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("Energy")
        ax.grid(True)
        
        # Mark minima and saddle points if they're within the cross-section
        minima = self.get_minima()
        
        # Filter points that are on or very close to our cross-section
        epsilon = 1e-6
        for point in minima:
            if abs(point[1].item() - y_value) < epsilon:
                ax.plot(point[0].item(), self.energy(point).item(), 'ro', markersize=8, label='Minimum')
        
        if hasattr(self, "get_true_transition_states"):
            saddle_points = self.get_true_transition_states()
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


class BaseGADEnergyFunction():
    def __init__(
        self, 
        gad_offset=100.0, clip_energy=True, stitching=True, stop_grad_ev=False, 
        div_epsilon=1e-12, clamp_min=None, clamp_max=None,
        *args,
        **kwargs
        ):
        self.gad_offset = gad_offset
        self.clip_energy = clip_energy
        self.stitching = stitching
        self.stop_grad_ev = stop_grad_ev
        self.div_epsilon = div_epsilon
        self.clamp_min = float(clamp_min) if clamp_min is not None else None
        self.clamp_max = float(clamp_max) if clamp_max is not None else None
        
    def compute_gad_potential(self, get_energy, samples):
        #####################################################################
        # Compute energy
        energy = get_energy(samples)

        #####################################################################
        # Compute forces

        # Use functorch.grad to compute forces
        try:
            # if len(samples.shape) == 1:
            #     forces = torch.func.grad(get_energy)(samples)
            # else:
            forces = torch.vmap(torch.func.grad(get_energy))(samples)
        except Exception as e:
            print(f"Samples: {samples}")
            print(f"Energy: {energy}")
            with open("gad_nan_log.txt", "a") as f:
                f.write(traceback.format_exc())
                f.write(f"Epoch: {self.curr_epoch}\n")
                f.write(f"Samples: {samples}\n")
                f.write(f"Energy: {energy}\n")
                f.write("-" * 80 + "\n")
            raise e

        #####################################################################
        # Compute two smallest eigenvalues of Hessian
        # if len(samples.shape) == 1:
        #     # Handle single sample
        #     hessian = torch.func.hessian(get_energy)(samples)
        #     eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        #     # Sort eigenvalues and corresponding eigenvectors
        #     sorted_indices = torch.argsort(eigenvalues)
        #     eigenvalues = eigenvalues[sorted_indices]
        #     eigenvectors = eigenvectors[:, sorted_indices]
        #     # Get 2 smallest eigenvalues and their eigenvectors
        #     smallest_eigenvalues = eigenvalues[:2]
        #     smallest_eigenvectors = eigenvectors[:, :2]
        # else:
        # Handle batched inputs using vmap # [B, D, D]
        batched_hessian = torch.vmap(torch.func.hessian(get_energy))(samples)
        # Get eigenvalues and eigenvectors for each sample in batch
        batched_eigenvalues, batched_eigenvectors = torch.linalg.eigh(
            batched_hessian
        )
        # Sort eigenvalues in ascending order and get corresponding indices
        sorted_indices = torch.argsort(batched_eigenvalues, dim=-1)
        # Get sorted eigenvalues
        batched_eigenvalues = torch.gather(batched_eigenvalues, -1, sorted_indices)
        # Get 2 smallest eigenvalues for each sample
        smallest_eigenvalues = batched_eigenvalues[..., :2]  # [B, 2]
        # Get eigenvectors corresponding to eigenvalues
        smallest_eigenvectors = torch.gather(
            batched_eigenvectors,
            -1,
            sorted_indices[..., 0:1]
            .unsqueeze(-1)
            .expand(batched_eigenvectors.shape),
        )

        if self.stop_grad_ev:
            smallest_eigenvalues = smallest_eigenvalues.detach()
            smallest_eigenvectors = smallest_eigenvectors.detach()

        # Get smallest eigenvalue and the corresponding eigenvector for each sample
        smallest_eigenvector = smallest_eigenvectors[..., 0]
        smallest_eigenvalue = smallest_eigenvalues[..., 0]
        
        #####################################################################
        # Compute GAD energy

        # stitching
        if self.stitching:
            if self.clip_energy:
                # in Luca's example the double well is [0, 20], gad_offset=50
                # here the dw is [-10, 170]
                pseudo_energy = torch.where(
                    smallest_eigenvalue < 0, 
                    input=torch.clip(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                        + self.gad_offset,
                        min=self.clamp_min,
                        max=self.clamp_max,
                    ), 
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1]
                )
            else:
                pseudo_energy = torch.where(
                    smallest_eigenvalue < 0, 
                    input=(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                    ), 
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1]
                )
        else:
            if self.clip_energy:
                pseudo_energy =(-energy
                    + (1 / (smallest_eigenvalue + self.div_epsilon))
                    * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                    + self.gad_offset)
                print(f"pseudo_energy: min={pseudo_energy.min():.1f}, max={pseudo_energy.max():.1f}")
                pseudo_energy = torch.clip(
                    pseudo_energy,
                    min=self.clamp_min,
                    max=self.clamp_max,
                )
                print(f"pseudo_energy clipped: min={pseudo_energy.min():.1f}, max={pseudo_energy.max():.1f}")
            else:
                pseudo_energy = (
                    -energy
                    + (1 / (smallest_eigenvalue + self.div_epsilon))
                    * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                )
                
        aux_output = {
            "energy": energy,
            "forces": forces,
            "smallest_eigenvalues": smallest_eigenvalues,
            "smallest_eigenvectors": smallest_eigenvectors,
            "pseudo_energy": pseudo_energy,
        }
        return pseudo_energy, aux_output

class BasePseudoEnergyFunction():
    def __init__(
        self, 
        energy_weight=1.0,
        force_weight=1.0,
        forces_norm=None,
        force_exponent=1,
        force_exponent_eps=1e-6,
        force_activation=None,
        force_scale=1.0,
        hessian_weight=1.0,
        hessian_eigenvalue_penalty="softplus",
        hessian_scale=10.0,
        term_aggr="sum",
        use_vmap=True,
        *args,
        **kwargs
        ):
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.forces_norm = forces_norm
        self.force_exponent = force_exponent
        self.force_exponent_eps = force_exponent_eps
        self.force_activation = force_activation
        self.force_scale = force_scale
        self.hessian_weight = hessian_weight
        self.hessian_eigenvalue_penalty = hessian_eigenvalue_penalty
        self.hessian_scale = hessian_scale
        self.use_vmap = use_vmap
        self.term_aggr = term_aggr
    
    # def log_prob(
    #     self, samples: torch.Tensor, return_aux_output: bool = False
    # ) -> torch.Tensor:
    def compute_pseudo_potential(self, get_energy, samples):
        """Compute unnormalized pseudo-energy.

        Args:
            samples: Input positions tensor of shape (dimensionality,)

        Returns:
            Pseudo-energy value (scalar), aux_output: auxiliary outputs (dict)
        """

        # Compute energy (all positive after -log_prob)
        if self.energy_weight > 0:
            energy = self._energy(samples)
        else:
            energy = torch.zeros_like(samples[:, 0])

        # Compute force penalty
        if self.force_weight > 0:

            # Use functorch.grad to compute forces
            if len(samples.shape) == 1:
                forces = torch.func.grad(self._energy)(samples)
            else:
                forces = torch.vmap(torch.func.grad(self._energy))(samples)

            # Compute force magnitude
            force_magnitude = torch.linalg.norm(forces, ord=self.forces_norm, dim=-1)
            if self.force_exponent > 0:
                force_magnitude = force_magnitude**self.force_exponent
            else:
                force_magnitude = -(
                    (force_magnitude + self.force_exponent_eps) ** self.force_exponent
                )
            # force_magnitude += 1. # [0, inf] -> [1, inf]
            if self.force_activation == "tanh":
                force_magnitude = torch.tanh(force_magnitude * self.force_scale)
            elif self.force_activation == "sigmoid":
                force_magnitude = torch.sigmoid(force_magnitude * self.force_scale)
            elif self.force_activation == "softplus":
                force_magnitude = torch.nn.functional.softplus(
                    force_magnitude * self.force_scale
                )
            elif self.force_activation in [None, False]:
                pass
            else:
                raise ValueError(f"Invalid force_activation: {self.force_activation}")
        else:
            force_magnitude = torch.zeros_like(energy)

        # Compute two smallest eigenvalues of Hessian
        if self.hessian_weight > 0:
            if len(samples.shape) == 1:
                # Handle single sample
                hessian = torch.func.hessian(self._energy)(samples)
                eigenvalues = torch.linalg.eigvalsh(hessian)
                smallest_eigenvalues = torch.sort(eigenvalues, dim=-1)[0][:2]
            else:
                # Handle batched inputs using vmap # [B, D, D]
                batched_hessian = torch.vmap(torch.func.hessian(self._energy))(
                    samples
                )
                # Get eigenvalues for each sample in batch # [B, D]
                batched_eigenvalues = torch.linalg.eigvalsh(batched_hessian)
                # Sort eigenvalues in ascending order for each sample # [B, D]
                batched_eigenvalues = torch.sort(batched_eigenvalues, dim=-1)[0]
                # Get 2 smallest eigenvalues for each sample
                smallest_eigenvalues = batched_eigenvalues[..., :2]  # [B, 2]

            # Bias toward index-1 saddle points:
            # - First eigenvalue should be negative (minimize positive values)
            # - Second eigenvalue should be positive (minimize negative values)
            if self.hessian_eigenvalue_penalty == "softplus":
                # Using softplus which is differentiable everywhere but still creates one-sided penalties
                # if first eigenvalue > 0, increase energy
                ev1_bias = torch.nn.functional.softplus(smallest_eigenvalues[:, 0])
                # if second eigenvalue > 0, increase energy
                ev2_bias = torch.nn.functional.softplus(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "relu":
                ev1_bias = torch.relu(smallest_eigenvalues[:, 0])
                ev2_bias = torch.relu(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            # elif self.hessian_eigenvalue_penalty == 'heaviside':
            #     # 1 if smallest_eigenvalues[0] > 0 else 0
            #     ev1_bias = torch.heaviside(smallest_eigenvalues[:, 0], torch.tensor(0.))
            #     # 1 if smallest_eigenvalues[1] < 0 else 0
            #     ev2_bias = torch.heaviside(-smallest_eigenvalues[:, 1], torch.tensor(0.))
            #     saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "sigmoid_individual":
                ev1_bias = torch.sigmoid(smallest_eigenvalues[:, 0])
                ev2_bias = torch.sigmoid(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "mult":
                # Penalize if both eigenvalues are positive or negative
                ev1_bias = smallest_eigenvalues[:, 0]
                ev2_bias = smallest_eigenvalues[:, 1]
                saddle_bias = ev1_bias * ev2_bias
            elif self.hessian_eigenvalue_penalty == "sigmoid":
                saddle_bias = torch.sigmoid(
                    smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1] * self.hessian_scale
                )
            elif self.hessian_eigenvalue_penalty == "tanh":
                saddle_bias = torch.tanh(
                    smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1] * self.hessian_scale
                )
                # saddle_bias = -torch.nn.functional.softplus(
                #     -saddle_bias - 2.0
                # )  # [0, 1]
                saddle_bias += 1.0  # [1, 2]
            elif self.hessian_eigenvalue_penalty == "and":
                saddle_bias = torch.nn.functional.softplus(
                    smallest_eigenvalues[:, 0]
                    * smallest_eigenvalues[:, 1]
                    * self.hessian_scale
                )  # [0, inf], 0 if good
                saddle_bias = torch.tanh(saddle_bias)  # [0, 1]
                # saddle_bias = 1 - saddle_bias # [0, 1] -> [1, 0] # 1 if good
            elif self.hessian_eigenvalue_penalty == "tanh_mult":
                # Penalize if both eigenvalues are positive or negative
                # tanh: ~ -1 for negative, 1 for positive
                # both neg -> 1, both pos -> 1, one neg one pos -> 0
                ev1_bias = torch.tanh(smallest_eigenvalues[:, 0])
                ev2_bias = torch.tanh(smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias * ev2_bias  # [-1, 1]
                # saddle_bias += 1. # [0, 2]
                # saddle_bias = -torch.nn.functional.softplus(-saddle_bias-2.) # [0, 1]
                # saddle_bias += 1. # [1, 2]
            else:
                raise ValueError(
                    f"Invalid penalty function: {self.hessian_eigenvalue_penalty}"
                )
        else:
            # No penalty
            saddle_bias = torch.zeros_like(energy)

        # idx = torch.randperm(samples.shape[0])[:15]
        # for i in idx:
        #     print(f"evs={smallest_eigenvalues[i].tolist()} -> {saddle_bias[i]:.2f}")

        # ensure we have one value per batch
        assert (
            energy.shape == force_magnitude.shape == saddle_bias.shape
        ), f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"
        assert (
            energy.shape[0] == samples.shape[0]
        ), f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"

        # Combine loss terms
        energy_loss = self.energy_weight * energy
        force_loss = self.force_weight * force_magnitude
        hessian_loss = self.hessian_weight * saddle_bias

        if self.term_aggr == "sum":
            pseudo_energy = energy_loss + force_loss + hessian_loss
        elif "norm" in self.term_aggr:  # "2_norm"
            # p‑Norms provide a way to interpolate between simple addition (p=1) and the maximum function (as p→∞)
            p = float(self.term_aggr.split("_")[0])
            pseudo_energy = (energy_loss**p + force_loss**p + hessian_loss**p) ** (1 / p)
        elif self.term_aggr == "logsumexp":
            # Smooth Maximum via Log‑Sum‑Exp
            pseudo_energy = torch.logsumexp(
                torch.stack([energy_loss, force_loss, hessian_loss], dim=-1), dim=-1
            )
        elif self.term_aggr == "mult":
            pseudo_energy = energy_loss * force_loss * hessian_loss
        elif self.term_aggr == "multfh":
            # multiply acts like an `and` operation
            pseudo_energy = energy_loss + (force_loss * hessian_loss)
        elif self.term_aggr == "1mmultfh":
            # multiply acts like an `and` operation
            # pseudo_energy = 1 - energy_loss + (force_loss * hessian_loss)
            pseudo_energy = energy_loss + 1 - ((1 - force_loss) * (1 - hessian_loss))
            # pseudo_energy += 2. # [0, inf] -> [1, inf]
        else:
            raise ValueError(f"Invalid term_aggr: {self.term_aggr}")

        aux_output = {
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "hessian_loss": hessian_loss,
        }
        return pseudo_energy, aux_output