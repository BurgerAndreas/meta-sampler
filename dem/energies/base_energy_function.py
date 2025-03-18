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
        
    def log_datasets(
        self,
        wandb_logger: Optional[WandbLogger] = None,
        n_contour_levels: int = 50,
        plotting_bounds: Optional[tuple] = None,
        plot_minima: bool = True,
        grid_width_n_points: int = 200,
        prefix: str = "",
        return_fig: bool = False,
    ) -> None:
        """Logs visualizations of test, validation, and training datasets.

        Args:
            wandb_logger (WandbLogger, optional): Logger for visualizations. Defaults to None.
            n_contour_levels (int, optional): Number of contour levels for plots. Defaults to 50.
            plotting_bounds (tuple, optional): Plot bounds. Defaults to None.
            plot_minima (bool, optional): Whether to plot minima. Defaults to True.
            grid_width_n_points (int, optional): Grid width for plotting. Defaults to 200.
            prefix (str, optional): Prefix for logged metrics. Defaults to "".
        """
        if wandb_logger is None and not return_fig:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        # Plot test set if available
        if self.test_set is not None:
            test_fig = self.get_single_dataset_fig(
                self.test_set,
                name="Test Set",
                n_contour_levels=n_contour_levels,
                plotting_bounds=plotting_bounds,
                plot_minima=plot_minima,
                grid_width_n_points=grid_width_n_points,
            )
            wandb_logger.log_image(
                f"{prefix}test_set", [test_fig]
            )

        # Plot validation set if available
        if self.val_set is not None:
            val_fig = self.get_single_dataset_fig(
                self.val_set,
                name="Validation Set",
                n_contour_levels=n_contour_levels,
                plotting_bounds=plotting_bounds,
                plot_minima=plot_minima,
                grid_width_n_points=grid_width_n_points,
            )
            wandb_logger.log_image(
                f"{prefix}val_set", [val_fig]
            )

        # Plot training set if available
        if self.train_set is not None:
            train_fig = self.get_single_dataset_fig(
                self.train_set,
                name="Training Set",
                n_contour_levels=n_contour_levels,
                plotting_bounds=plotting_bounds,
                plot_minima=plot_minima,
                grid_width_n_points=grid_width_n_points,
            )
            wandb_logger.log_image(
                f"{prefix}train_set", [train_fig]
            )
        # if return_fig:
        #     return fig
        # else:
        #     return fig_to_image(fig)

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
        plot_transition_states=False,
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
            
        if plot_transition_states and hasattr(self, "get_true_transition_states"):
            transition_states = self.get_true_transition_states()
            ax.scatter(*transition_states.detach().cpu().T, color="green", marker="o")

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
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Evaluates the normalized (pseudo-)energy function at given samples.

        Args:
            samples (torch.Tensor): Input points

        Returns:
            torch.Tensor: Energy values at input points
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return self.log_prob(
            samples, temperature=temperature, return_aux_output=return_aux_output
        )

    def log_prob(
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Evaluates the unnormalized log probability of the (pseudo-)energy function at given samples.

        Args:
            samples (torch.Tensor): Input points
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False

        Returns:
            torch.Tensor: Log probability values at input points
        """
        raise NotImplementedError(
            f"Energy function {self.__class__.__name__} must implement `log_prob()`"
        )

    def _energy(
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Energy of the unnormalized (physical) potential. Used in GAD to compute forces/Hessian.

        Args:
            samples (torch.Tensor): Input points
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False

        Returns:
            torch.Tensor: Energy values at input points
        """
        raise NotImplementedError(
            f"GAD energy function {self.__class__.__name__} must implement `_energy()`"
        )

    def energy(
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Energy=-logprob of unnormalized (pseudo-)potential.
        Convinience fuction. Superflous with `log_prob()`. Used for plotting.
        """
        if return_aux_output:
            log_prob, aux_output = self.log_prob(
                samples, temperature=temperature, return_aux_output=True
            )
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
                assert torch.isfinite(
                    latest_samples
                ).all(), f"Max value: {latest_samples.max()}, Min value: {latest_samples.min()}"
                plt.close()
                fig, ax = plt.subplots()
                ax.scatter(*latest_samples.detach().cpu().T)
                plt.xlim(self._plotting_bounds[0], self._plotting_bounds[1])
                plt.ylim(self._plotting_bounds[0], self._plotting_bounds[1])

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
        batched_eigenvalues, batched_eigenvectors = torch.linalg.eigh(hessian)

        # Sort eigenvalues
        sorted_indices = torch.argsort(batched_eigenvalues, dim=-1)
        batched_eigenvalues = torch.gather(batched_eigenvalues, -1, sorted_indices)
        # Need to expand sorted_indices to match eigenvectors dimensions
        # The error occurs because eigenvectors have shape [..., dim, dim] while indices are [..., dim]
        expanded_indices = sorted_indices.unsqueeze(-1).expand(
            *batched_eigenvalues.shape, self._dimensionality
        )
        batched_eigenvectors = torch.gather(batched_eigenvectors, -2, expanded_indices)

        # Reshape for plotting
        eigenvalues_grid = batched_eigenvalues.reshape(
            grid_width_n_points, grid_width_n_points, self._dimensionality
        )
        eigenvectors_grid = batched_eigenvectors.reshape(
            grid_width_n_points,
            grid_width_n_points,
            self._dimensionality,
            self._dimensionality,
        )
        x_grid = x_points_dim1.reshape(-1, 1).repeat(1, grid_width_n_points)
        y_grid = x_points_dim2.reshape(1, -1).repeat(grid_width_n_points, 1)

        return x_grid, y_grid, eigenvalues_grid, eigenvectors_grid

    def plot_hessian_eigenvalues(
        self, grid_width_n_points=200, plotting_bounds=None, name=None, skip=10
    ):
        """Plot the first two eigenvalues of the Hessian on a grid.

        Args:
            grid_width_n_points (int): Number of points in each dimension for grid
            plotting_bounds (tuple, optional): Plot bounds as (min, max) tuple

        Returns:
            matplotlib.figure.Figure: Figure with eigenvalue plots
        """
        if self.dimensionality != 2:
            print(
                f"Hessian eigenvalues are only defined for 2D systems, but this system has {self.dimensionality} dimensions"
            )
            return

        x_grid, y_grid, eigenvalues_grid, eigenvectors_grid = (
            self.get_hessian_eigenvalues_on_grid(
                grid_width_n_points=grid_width_n_points, plotting_bounds=plotting_bounds
            )
        )
        eigenvalues_grid = eigenvalues_grid.cpu().numpy()
        eigenvectors_grid = eigenvectors_grid.cpu().numpy()
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
        axs[0].quiver(
            x_grid[::skip, ::skip],
            y_grid[::skip, ::skip],
            eigenvectors_grid[::skip, ::skip, 0, 0],
            eigenvectors_grid[::skip, ::skip, 0, 1],
            color="white",
            alpha=0.8,
        )
        axs[0].set_title("First (smallest) eigenvalue")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(im1, ax=axs[0])

        # Plot second eigenvalue
        im2 = axs[1].pcolormesh(
            x_grid, y_grid, eigenvalues_grid[:, :, 1], cmap="viridis", shading="auto"
        )
        axs[1].quiver(
            x_grid[::skip, ::skip],
            y_grid[::skip, ::skip],
            eigenvectors_grid[::skip, ::skip, 1, 0],
            eigenvectors_grid[::skip, ::skip, 1, 1],
            color="white",
            alpha=0.8,
        )
        axs[1].set_title("Second eigenvalue")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()

        return fig

    def plot_gradient(
        self,
        grid_width_n_points=200,
        skip=10,
        plotting_bounds=None,
        name=None,
    ):
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
            print(
                f"Gradient plotting is only defined for 2D systems, but this system has {self.dimensionality} dimensions"
            )
            return

        device = self.device

        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds

        N = grid_width_n_points
        # Create a 2D grid
        xgrid, ygrid = torch.meshgrid(
            torch.linspace(plotting_bounds[0], plotting_bounds[1], N, device=device),
            torch.linspace(plotting_bounds[0], plotting_bounds[1], N, device=device),
            indexing="ij",
        )
        grid_flat = torch.stack([xgrid.flatten(), ygrid.flatten()], axis=1)

        def V(x):
            return -self.log_prob(x).squeeze(0)

        V_vmap = torch.vmap(V)
        # V_vmap = torch.jit.script(V_vmap) # doesn't work with self.
        # V = torch.jit.script(V)

        # Calculate gradient using JAX for a single point
        def grad_V(x):
            return -torch.func.grad(V)(x)

        grad_V_vmap = torch.vmap(grad_V)
        # grad_V_vmap = torch.jit.script(grad_V_vmap)

        # Calculate the potential values
        Z = V_vmap(grid_flat).reshape(N, N)

        # Calculate gradients
        V_grad = grad_V_vmap(grid_flat).reshape(N, N, 2)

        # move tensors to cpu
        xgrid = xgrid.cpu().numpy()
        ygrid = ygrid.cpu().numpy()
        Z = Z.cpu().numpy()
        V_grad = V_grad.cpu().numpy()

        # Create the plot
        plt.close()
        plt.figure(figsize=(10, 8))
        # Create a contour plot
        plt.contour(xgrid, ygrid, Z, levels=20)
        plt.colorbar(label="Potential Energy")

        # Add filled contours for better visualization
        plt.contourf(xgrid, ygrid, Z, levels=20, alpha=0.7)

        # Add gradient field (using fewer points for clarity)
        plt.quiver(
            xgrid[::skip, ::skip],
            ygrid[::skip, ::skip],
            V_grad[::skip, ::skip, 0],
            V_grad[::skip, ::skip, 1],
            color="white",
            alpha=0.8,
        )

        # Add minima points if available
        if hasattr(self, "get_minima"):
            minima = self.get_minima()
            if minima is not None:
                plt.scatter(
                    *minima.detach().cpu().T,
                    color="red",
                    marker="x",
                    s=100,
                    label="Minima",
                )
                plt.legend()

        plt.tight_layout()

        return plt.gcf()

    def plot_energy_crossection(
        self, n_points=200, plotting_bounds=None, y_value=0.0, name=None
    ):
        """Plot a horizontal cross-section of the energy landscape at a specified y-value.

        Args:
            n_points (int): Number of points to sample along the x-axis
            y_value (float): The y-value at which to take the cross-section

        Returns:
            matplotlib.figure.Figure: Figure with the energy cross-section plot
        """
        if self.dimensionality != 2:
            print(
                f"Energy cross-section is only defined for 2D systems, but this system has {self.dimensionality} dimensions"
            )
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
                ax.plot(
                    point[0].item(),
                    self.energy(point).item(),
                    "ro",
                    markersize=8,
                    label="Minimum",
                )

        if hasattr(self, "get_true_transition_states"):
            saddle_points = self.get_true_transition_states()
            for point in saddle_points:
                if abs(point[1].item() - y_value) < epsilon:
                    ax.plot(
                        point[0].item(),
                        self.energy(point).item(),
                        "go",
                        markersize=8,
                        label="Saddle point",
                    )

        # Add legend if we plotted any special points
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        plt.tight_layout()

        # Restore device
        self.move_to_device(self.device)

        return fig

    def plot_energy_crossection_along_axis(
        self, n_points=200, plotting_bounds=None, axis=0, axis_value=0.0, name=None
    ):
        """Plot a horizontal cross-section of the energy landscape at a specified y-value.

        Args:
            n_points (int): Number of points to sample along the x-axis
            y_value (float): The y-value at which to take the cross-section

        Returns:
            matplotlib.figure.Figure: Figure with the energy cross-section plot
        """
        if self.dimensionality != 2:
            print(
                f"Energy cross-section is only defined for 2D systems, but this system has {self.dimensionality} dimensions"
            )
            return

        other_axis = 1 if axis == 0 else 0
        axissymbol = "x" if axis == 0 else "y"

        self.move_to_device("cpu")

        # Create x points for the cross-section
        if plotting_bounds is None:
            plotting_bounds = self._plotting_bounds
        axis_points = torch.linspace(plotting_bounds[0], plotting_bounds[1], n_points)

        # Create samples with fixed value
        samples = torch.zeros((n_points, 2))
        samples[:, axis] = axis_value
        samples[:, other_axis] = axis_points

        # Compute energy values
        energy_values = self.energy(samples).detach().cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot energy cross-section
        ax.plot(axis_points, energy_values)
        title = f"Energy cross-section at {axissymbol}={axis_value:.1f}"
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
            if abs(point[axis].item() - axis_value) < epsilon:
                ax.plot(
                    point[other_axis].item(),
                    self.energy(point).item(),
                    "ro",
                    markersize=8,
                    label="Minimum",
                )

        # if hasattr(self, "get_true_transition_states"):
        #     saddle_points = self.get_true_transition_states()
        #     for point in saddle_points:
        #         if abs(point[other_axis].item() - axis_value) < epsilon:
        #             ax.plot(
        #                 point[axis].item(),
        #                 self.energy(point).item(),
        #                 "go",
        #                 markersize=8,
        #                 label="Saddle point",
        #             )

        # Add legend if we plotted any special points
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

        plt.tight_layout()

        # Restore device
        self.move_to_device(self.device)

        return fig


class BasePseudoEnergyFunction:
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
        gad_offset=10.0,
        clamp_min=None,
        clamp_max=None,
        use_vmap=True,
        dataset_noise_scale=0.0,
        # GAD parameters
        clip_energy=True,
        stitching=True,
        stop_grad_ev=False,
        div_epsilon=1e-12,
        *args,
        **kwargs,
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
        self.gad_offset = gad_offset
        self.clamp_min = float(clamp_min) if clamp_min is not None else None
        self.clamp_max = float(clamp_max) if clamp_max is not None else None
        self.dataset_noise_scale = dataset_noise_scale
        # GAD parameters
        self.clip_energy = clip_energy
        self.stitching = stitching
        self.stop_grad_ev = stop_grad_ev
        self.div_epsilon = div_epsilon
        
        self.transition_points = None
        
    def compute_pseudo_potential(self, get_energy, samples):
        # TODO: clean up this mess
        if self.term_aggr.lower() == "gad":
            return self._compute_gad_potential(get_energy, samples)
        else:
            return self._compute_pseudo_potential(get_energy, samples)
        
    def _compute_gad_potential(self, get_energy, samples):
        #####################################################################
        # Compute energy
        energy = get_energy(samples)

        #####################################################################
        # Compute forces
        forces = torch.vmap(torch.func.grad(get_energy))(samples)

        #####################################################################
        # Compute two smallest eigenvalues of Hessian
        # Handle batched inputs using vmap # [B, D, D]
        batched_hessian = torch.vmap(torch.func.hessian(get_energy))(samples)
        # Get eigenvalues and eigenvectors for each sample in batch
        batched_eigenvalues, batched_eigenvectors = torch.linalg.eigh(batched_hessian)
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
            sorted_indices[..., 0:1].unsqueeze(-1).expand(batched_eigenvectors.shape),
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
                    # smallest_eigenvalue < 0,
                    (smallest_eigenvalue * smallest_eigenvalues[..., 1]) < 0,
                    input=torch.clip(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                        + self.gad_offset,
                        min=self.clamp_min,
                        max=self.clamp_max,
                    ),
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1],
                )
            else:
                pseudo_energy = torch.where(
                    # smallest_eigenvalue < 0,
                    (smallest_eigenvalue * smallest_eigenvalues[..., 1]) < 0,
                    input=(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                    ),
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1],
                )
        else:
            if self.clip_energy:
                pseudo_energy = torch.clip(
                    -energy
                    + (1 / (smallest_eigenvalue + self.div_epsilon))
                    * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                )
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

    def _compute_pseudo_potential(self, get_energy, samples):
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
                batched_hessian = torch.vmap(torch.func.hessian(self._energy))(samples)
                # Get eigenvalues and eigenvectors for each sample in batch
                batched_eigenvalues, batched_eigenvectors = torch.linalg.eigh(
                    batched_hessian
                )
                # Sort eigenvalues in ascending order and get corresponding indices
                sorted_indices = torch.argsort(batched_eigenvalues, dim=-1)
                # Get sorted eigenvalues
                batched_eigenvalues = torch.gather(
                    batched_eigenvalues, -1, sorted_indices
                )
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

            # Bias toward index-1 saddle points:
            # - First eigenvalue should be negative (minimize positive values)
            # - Second eigenvalue should be positive (minimize negative values)
            if self.hessian_eigenvalue_penalty in [None, False]:
                saddle_bias = smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1]
            elif self.hessian_eigenvalue_penalty == "softplus":
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
                    smallest_eigenvalues[:, 0]
                    * smallest_eigenvalues[:, 1]
                    * self.hessian_scale
                )
            elif self.hessian_eigenvalue_penalty == "tanh":
                saddle_bias = torch.tanh(
                    smallest_eigenvalues[:, 0]
                    * smallest_eigenvalues[:, 1]
                    * self.hessian_scale
                )
                # saddle_bias = -torch.nn.functional.softplus(
                #     -saddle_bias - 2.0
                # )  # [0, 1]
                saddle_bias = saddle_bias + 1.0  # [1, 2]
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
            pseudo_energy = (
                energy_loss**p + force_loss**p + hessian_loss**p
            ) ** (1 / p)
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
        elif self.term_aggr == "cond_force":
            pseudo_energy = torch.where(
                # smallest_eigenvalue < 0,
                saddle_bias < 0,
                input=torch.clip(
                    force_magnitude
                    # torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 0]) ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                ),
                other=saddle_bias,
            )
        elif self.term_aggr == "cond_force_proj":
            pseudo_energy = torch.where(
                # smallest_eigenvalue < 0,
                saddle_bias < 0,
                input=torch.clip(
                    # force_magnitude
                    torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 0]) ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                ),
                other=saddle_bias,
            )
        elif self.term_aggr == "cond_force_proj2":
            pseudo_energy = torch.where(
                # smallest_eigenvalue < 0,
                saddle_bias < 0,
                input=torch.clip(
                    # force_magnitude
                    torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 1]) ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                ),
                other=saddle_bias,
            )
        elif self.term_aggr == "cond_force_proj_both":
            pseudo_energy = torch.where(
                # smallest_eigenvalue < 0,
                saddle_bias < 0,
                input=torch.clip(
                    # force_magnitude
                    torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 0]) ** 2
                    + torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 1])
                    ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                ),
                other=saddle_bias,
            )
        else:
            raise ValueError(f"Invalid term_aggr: {self.term_aggr}")

        aux_output = {
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "hessian_loss": hessian_loss,
        }
        return pseudo_energy, aux_output

    def _setup_dataset(self, n_samples):
        """Return the true transition states as the test set.
        
        Duplicates entries to match the number of requested samples.
        Optionally adds a small amount of Gaussian noise to prevent exact duplicates.
        """
        # Get the true transition states
        transition_states = self.get_true_transition_states()
        
        # If no transition states found, return empty tensor
        if len(transition_states) == 0:
            return torch.tensor([], device=self.device)
            
        # Calculate how many times to repeat each transition state
        n_states = len(transition_states)
        repeats = max(1, n_samples // n_states)
        
        # Duplicate the transition states to match requested sample count
        duplicated_states = transition_states.repeat(repeats, 1)
        
        # Optionally add small Gaussian noise to prevent exact duplicates
        if self.dataset_noise_scale > 0:
            noise = torch.randn_like(duplicated_states) * self.dataset_noise_scale
            duplicated_states = duplicated_states + noise
            
        # Trim to exact number of samples if needed
        duplicated_states = duplicated_states[:n_samples]
            
        return duplicated_states
    
    def setup_val_set(self):
        return self._setup_dataset(self.val_set_size)
    
    def setup_test_set(self):
        return self._setup_dataset(self.test_set_size)
    
    def setup_train_set(self):
        return self._setup_dataset(self.train_set_size)
    
    
    # def find_stationary_point(self, initial_guess, method="hybr"):
    #     """
    #     Solve grad_neg_log_prob(x) = 0 starting from initial_guess.

    #     Args:
    #         initial_guess: np.ndarray of shape (dim,)
    #         gmm: instance of GMM class
    #         method: solver method (e.g., 'hybr', 'lm', etc.)
    #     Returns:
    #         result: scipy.optimize.OptimizeResult object
    #     """
    #     # Convert initial guess to numpy if it's a torch tensor
    #     if isinstance(initial_guess, torch.Tensor):
    #         initial_guess = initial_guess.detach().cpu().numpy()

    #     result = scipy.optimize.root(
    #         fun=grad_neg_log_prob, x0=initial_guess, args=(self,), method=method
    #     )
    #     return result
    
    # def get_true_transition_states(self, grid_size=200):
    #     """Find saddle points using scipy.optimize.root and Hessian eigenvalue analysis.

    #     Args:
    #         grid_size (int, optional): Number of points along each dimension
    #         bounds (tuple, optional): (min, max) bounds for grid

    #     Returns:
    #         torch.Tensor: Coordinates of identified saddle points
    #     """
    #     if self.transition_points is not None:
    #         return self.transition_points

    #     fname = f"dem_outputs/transition_points_gmm{self.gmm.n_mixes}.npy"
    #     if os.path.exists(fname):
    #         self.transition_points = torch.tensor(np.load(fname), device=self.device)
    #         # print(f"Loaded transition points from {fname}")
    #         return self.transition_points
        
    #     starting_points = torch.meshgrid(
    #         torch.linspace(self.plotting_bounds[0], self.plotting_bounds[1], grid_size),
    #         torch.linspace(self.plotting_bounds[0], self.plotting_bounds[1], grid_size),
    #     )
    #     starting_points = starting_points.reshape(-1, 2)
    # for each starting point, find the stationary point
    #     transition_points = []
    #     for point in starting_points:
    #         result = self.find_stationary_point(point)
    #         if result.success:
    #             transition_points.append(result.x)
    #     transition_points = torch.tensor(transition_points, device=self.device)
        
    #     # Validate that this is indeed a saddle point by checking:
    #     # 1. The gradient (force) should be close to zero
    #     # 2. The Hessian should have exactly one negative eigenvalue
        
    #     # Set up for gradient and Hessian computation
    #     x = transition_state.clone().requires_grad_(True)
        
    #     # Compute the energy (negative log probability)
    #     def neg_log_prob(x):
    #         return -self.log_prob(x)
        
    #     # Compute gradient (force)
    #     grad = torch.func.grad(neg_log_prob)(x)
        
    #     # Compute Hessian
    #     hessian = torch.func.hessian(neg_log_prob)(x)
        
    #     # Check eigenvalues
    #     eigenvals, _ = torch.linalg.eigh(hessian)
        
    #     # Verify this is a saddle point (index-1)
    #     grad_norm = grad.norm().item()
    #     n_negative = torch.sum(eigenvals < -1e-6).item()
        
    #     print(f"Transition state: {transition_state.tolist()}")
    #     print(f"Gradient norm: {grad_norm:.6f}")
    #     print(f"Hessian eigenvalues: {eigenvals.tolist()}")
    #     print(f"Number of negative eigenvalues: {n_negative}")
        
    #     # Store the validated transition state
    #     self.transition_points = transition_state
        
    #     # Save to file for future use
    #     os.makedirs("dem_outputs", exist_ok=True)
    #     np.save(fname, self.transition_points.detach().cpu().numpy())
        
    #     return self.transition_points