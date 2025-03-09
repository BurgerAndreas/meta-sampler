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


class BaseEnergyFunction(ABC):
    """Base class for energy functions used in DEM.

    This class provides the basic interface and functionality for energy functions,
    including data normalization, sampling from datasets, and logging.

    Args:
        dimensionality (int): Dimension of the input space
        is_molecule (bool, optional): Whether this energy function is for a molecule. Defaults to False.
        normalization_min (float, optional): Minimum value for normalization. Defaults to None.
        normalization_max (float, optional): Maximum value for normalization. Defaults to None.
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
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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

        return fig_to_image(fig)

    def log_prob(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the log probability of the energy function at given samples.

        Args:
            samples (torch.Tensor): Input points
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False

        Returns:
            torch.Tensor: Log probability values at input points
        """
        raise NotImplementedError

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

    @abstractmethod
    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the energy function at given samples.

        Args:
            samples (torch.Tensor): Input points

        Returns:
            torch.Tensor: Energy values at input points
        """
        raise NotImplementedError

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
