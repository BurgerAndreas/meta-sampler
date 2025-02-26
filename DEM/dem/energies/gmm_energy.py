from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot after setting backend

plt.ioff()  # Turn off interactive model

import numpy as np
import torch
import fab.target_distributions
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image
from dem.utils.plotting import plot_fn, plot_marginal_pair


class GMMEnergy(BaseEnergyFunction):
    """Gaussian Mixture Model energy function for DEM.

    This class implements a GMM energy function that can be used for training and evaluating DEM models.
    Used in train.py and eval.py as the energy_function parameter.

    Args:
        dimensionality (int, optional): Dimension of input space. Defaults to 2.
        n_mixes (int, optional): Number of Gaussian components. Defaults to 40.
        loc_scaling (float, optional): Scale factor for component means. Defaults to 40.
        log_var_scaling (float, optional): Scale factor for component variances. Defaults to 1.0.
        device (str, optional): Device to place tensors on. Defaults to "cpu".
        true_expectation_estimation_n_samples (int, optional): Number of samples for expectation estimation. Defaults to 1e5.
        plotting_buffer_sample_size (int, optional): Number of samples to plot. Defaults to 512.
        plot_samples_epoch_period (int, optional): How often to plot samples. Defaults to 5.
        should_unnormalize (bool, optional): Whether to unnormalize samples. Defaults to False.
        data_normalization_factor (float, optional): Factor for data normalization. Defaults to 50.
        train_set_size (int, optional): Size of training set. Defaults to 100000.
        test_set_size (int, optional): Size of test set. Defaults to 2000.
        val_set_size (int, optional): Size of validation set. Defaults to 2000.
        data_path_train (str, optional): Path to training data file. Defaults to None.
    """

    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        loc_scaling=40,
        log_var_scaling=1.0,
        device="cpu",
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=50,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        self.gmm = fab.target_distributions.gmm.GMM(
            dim=dimensionality,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=use_gpu,
            true_expectation_estimation_n_samples=true_expectation_estimation_n_samples,
        )

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.data_path_train = data_path_train

        self.name = "gmm"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        """Sets up test dataset by sampling from GMM.
        Used in train.py and eval.py during model testing.

        Returns:
            torch.Tensor: Test dataset tensor
        """
        # test_sample = self.gmm.sample((self.test_set_size,))
        # return test_sample
        return self.gmm.test_set

    def setup_train_set(self):
        """Sets up training dataset by sampling from GMM or loading from file.
        Used in train.py during model training.

        Returns:
            torch.Tensor: Training dataset tensor
        """
        if self.data_path_train is None:
            train_samples = self.normalize(self.gmm.sample((self.train_set_size,)))

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
        val_samples = self.gmm.sample((self.val_set_size,))
        return val_samples

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates GMM log probability at given samples.
        Used in train.py and eval.py for computing model loss.

        Args:
            samples (torch.Tensor): Input points to evaluate

        Returns:
            torch.Tensor: Log probability values at input points
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        if return_aux_output:
            aux_output = {}
            return self.gmm.log_prob(samples), aux_output
        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        """Dimension of input space.
        Used in train.py and eval.py for model initialization.

        Returns:
            int: Input dimensionality
        """
        return 2

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
        return_fig: bool = False,
    ) -> None:
        """Logs metrics and visualizations at the end of each epoch.
        Used in train.py for logging training progress.

        Args:
            latest_samples (torch.Tensor): Most recent generated samples
            latest_energies (torch.Tensor): Energy values for latest samples
            wandb_logger (WandbLogger): Logger for metrics and visualizations
            unprioritized_buffer_samples (torch.Tensor, optional): Samples from unprioritized buffer
            cfm_samples (torch.Tensor, optional): Samples from CFM model
            replay_buffer (ReplayBuffer, optional): Replay buffer containing past samples
            prefix (str, optional): Prefix for logged metrics. Defaults to ""
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

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        """Logs sample visualizations.
        Used in train.py for logging sample distributions.

        Args:
            samples (torch.Tensor): Samples to visualize
            wandb_logger (WandbLogger): Logger for visualizations
            name (str, optional): Name for logged images. Defaults to ""
            should_unnormalize (bool, optional): Whether to unnormalize samples. Defaults to False
        """
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_single_dataset_fig(
        self,
        samples,
        name,
        n_contour_levels=50,
        plotting_bounds=(-1.4 * 40, 1.4 * 40),
        plot_gaussian_means=False,
        grid_width_n_points=200,
        plot_style="contours",
        with_legend=False,
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
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        plot_fn(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=grid_width_n_points,
        )
        if samples is not None:
            plot_marginal_pair(samples, ax=ax, bounds=plotting_bounds)
        if name is not None:
            ax.set_title(f"{name}")

        if plot_gaussian_means:
            means = self.gmm.distribution.component_distribution.loc
            ax.scatter(*means.detach().cpu().T, color="red", marker="x")
            # ax.legend()

        if with_legend:
            ax.legend()

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def get_dataset_fig(
        self,
        samples,
        gen_samples=None,
        n_contour_levels=50,
        plotting_bounds=(-1.4 * 40, 1.4 * 40),
        plot_gaussian_means=False,
        plot_style="contours",
        plot_prob_kwargs={},
        plot_sample_kwargs={},
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
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_fn(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=200,
            plot_kwargs=plot_prob_kwargs,
        )

        # plot dataset samples
        if samples is not None:
            plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
            axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_fn(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                plot_style=plot_style,
                n_contour_levels=50,
                grid_width_n_points=200,
                plot_kwargs=plot_prob_kwargs,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds, plot_kwargs=plot_sample_kwargs)
            axs[1].set_title("Generated samples")

        if plot_gaussian_means:
            means = self.gmm.distribution.component_distribution.loc
            axs[1].scatter(*means.detach().cpu().T, color="red", marker="x")
            # axs[1].legend()

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)
