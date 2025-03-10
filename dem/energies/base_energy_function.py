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
            # "forces": forces,
            # "smallest_eigenvalues": smallest_eigenvalues,
            # "smallest_eigenvectors": smallest_eigenvectors,
            # "pseudo_energy": pseudo_energy,
        }
        return pseudo_energy, aux_output
