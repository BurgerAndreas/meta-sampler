import numpy as np
import torch
from typing import Callable
from dem.energies.base_energy_function import BaseEnergyFunction
import matplotlib.pyplot as plt
import itertools
from typing import Optional, Dict, Any, Tuple
from dem.utils.plotting import plot_contours, plot_marginal_pair

# adapted from https://github.com/lollcat/fab-torch/blob/master/fab/target_distributions/double_well.py


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
        shift=0.5,  # shift the minima/transition state in the first dimension (x direction)
        device="cpu",
        is_molecule=False,
        true_expectation_estimation_n_samples=int(1e5),
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=1,
        train_set_size=0,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
        *args,
        **kwargs
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
        self.shift = shift
        if self._b == -6 and self._c == 1.0:
            self.means = torch.tensor([-1.7, 1.7], device=self.device)
            self.scales = torch.tensor([0.5, 0.5], device=self.device)
        else:
            raise NotImplementedError

        self.component_mix = self.compute_mc_component_mix()

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

    def compute_mc_component_mix(self):
        # Calculate component_mix based on bias parameter a
        # Calculate relative populations using Boltzmann distribution
        # p ∝ exp(-E/kT)
        # For relative populations, we can set kT = 1 (temperature scaling is handled in the energy function)
        # Get energies at the minima
        energy_min1 = self.energy(torch.tensor([[-1.7, 0]], device=self.device))
        energy_min2 = self.energy(torch.tensor([[1.7, 0]], device=self.device))
        # Calculate Boltzmann factors
        boltzmann_factor_min1 = torch.exp(-energy_min1)
        boltzmann_factor_min2 = torch.exp(-energy_min2)
        # Convert to mixture weights that sum to 1 = relative populations
        total_factor = boltzmann_factor_min1 + boltzmann_factor_min2
        population_min1 = boltzmann_factor_min1 / total_factor
        population_min2 = boltzmann_factor_min2 / total_factor
        return torch.tensor([population_min1, population_min2], device=self.device)

    def move_to_device(self, device):
        self.component_mix = self.component_mix.to(device)
        self.means = self.means.to(device)
        self.scales = self.scales.to(device)

    def log_prob(self, samples, temperature=1.0, return_aux_output=False):
        assert (
            samples.shape[-1] == self._dimensionality
        ), "`x` does not match `dimensionality`"
        log_prob = torch.squeeze(-self._energy(samples))
        log_prob = log_prob / temperature
        if return_aux_output:
            return log_prob, {}
        return log_prob

    def force(self, samples, temperature=1.0):
        samples = samples.requires_grad_(True)
        e = -self.log_prob(samples, temperature=temperature)
        return -torch.autograd.grad(e.sum(), samples)[0]

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

    def sample_first_dimension(self, shape):
        assert len(shape) == 1

        # see fab.sampling_methods.rejection_sampling_test.py
        # Define target.
        def target_log_prob(x):
            x = x + self.shift
            return -(self._a * x + self._b * x.pow(2) + self._c * x.pow(4))

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

    def sample(self, shape):
        dim1_samples = self.sample_first_dimension(shape)
        # mean and std of the second dimension
        dim2_samples = torch.distributions.Normal(
            torch.tensor(0.0).to(dim1_samples.device),
            torch.tensor(1.0).to(dim1_samples.device),
        ).sample(shape)
        # dim2_samples = self.sample_first_dimension(shape)
        return torch.stack([dim1_samples, dim2_samples], dim=-1)

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
        return torch.tensor(
            [[-1.7 - self.shift, 0.0], [1.7 - self.shift, 0.0]], device=self.device
        )

    def get_true_transition_states(self):
        return torch.tensor([[-self.shift, 0.0]], device=self.device)

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
            if self.train_set_size > 0:
                train_samples = self.normalize(self.sample((self.train_set_size,)))
            else:
                return None

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


if __name__ == "__main__":
    # Test that rejection sampling is work as desired.
    import matplotlib.pyplot as plt

    target = DoubleWellEnergy(2)

    # x_linspace = torch.linspace(-4, 4, 200)

    # Z_dim_1 = 11784.50927
    # samples = target.sample((10000,))
    # p_1 = torch.exp(-target._energy_dim_1(x_linspace))
    # # plot first dimension vs normalised log prob
    # plt.plot(x_linspace, p_1 / Z_dim_1, label="p_1 normalised")
    # plt.hist(samples[:, 0], density=True, bins=100, label="sample density")
    # plt.legend()
    # plt.show()

    # # Now dimension 2.
    # Z_dim_2 = (2 * torch.pi) ** 0.5
    # p_2 = torch.exp(-target._energy_dim_2(x_linspace))
    # # plot first dimension vs normalised log prob
    # plt.plot(x_linspace, p_2 / Z_dim_2, label="p_2 normalised")
    # plt.hist(samples[:, 1], density=True, bins=100, label="sample density")
    # plt.legend()
    # plt.show()

    plt.close()
    fig, ax = plt.subplots()
    plot_contours(
        target.log_prob,
        bounds=(-4, 4),
        grid_width=200,
        n_contour_levels=100,
        ax=ax,
    )
    samples = target.sample((1000,))
    plot_marginal_pair(samples, bounds=(-4, 4), marginal_dims=(0, 1), ax=ax)
    ax.set_title("samples")
    # plot minima
    minima = target.get_minima()
    ax.scatter(minima[:, 0].cpu(), minima[:, 1].cpu(), color="black", marker="x", s=100)
    # plot transition states
    transition_states = target.get_true_transition_states()
    ax.scatter(
        transition_states[:, 0].cpu(),
        transition_states[:, 1].cpu(),
        color="red",
        marker="x",
        s=100,
    )
    plt.savefig("double_well_potential.png")
    print("saved figure to double_well_potential.png")
