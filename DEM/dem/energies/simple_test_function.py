from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image


def f_prime(x):
    return 6 * x * (x**2 - 1)**2

def energy(x):
    # Energy defined as the absolute value of f'(x)
    return torch.abs(f_prime(x))

class SimpleTestFunction(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=1,
        device="cpu",
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=2.0,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
    ):
        if dimensionality != 1:
            raise ValueError("SimpleTestFunction only supports dimensionality=1")
            
        torch.manual_seed(0)
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period
        
        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor
        
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        
        self.curr_epoch = 0
        self.name = "simple_test"

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        samples = torch.linspace(-2, 2, self.test_set_size, device=self.device)
        return samples.unsqueeze(-1)

    def setup_train_set(self):
        samples = torch.rand(self.train_set_size, device=self.device) * 4 - 2
        return samples.unsqueeze(-1)

    def setup_val_set(self):
        samples = torch.linspace(-2, 2, self.val_set_size, device=self.device)
        return samples.unsqueeze(-1)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return -energy(samples.squeeze(-1))

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if latest_samples is not None:
                samples_fig = self.get_dataset_fig(latest_samples)
                wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

        self.curr_epoch += 1

    def get_dataset_fig(self, samples):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the normalized energy function
        x = torch.linspace(-2, 2, 1000, device=self.device)
        y = torch.exp(-energy(x))
        # Normalize using trapezoidal rule
        dx = x[1] - x[0]
        Z = torch.trapz(y, x)
        y = y / Z
        
        ax.plot(x.cpu(), y.cpu(), 'b-', label='Target Distribution')
        
        # Plot the histogram of samples
        if samples is not None:
            samples = samples.squeeze(-1)
            ax.hist(samples.cpu(), bins=50, density=True, alpha=0.5, color='r', label='Samples')
        
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)
        
        return fig_to_image(fig) 