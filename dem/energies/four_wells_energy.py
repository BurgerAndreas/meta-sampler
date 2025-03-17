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


class FourWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        a=1.0,  # Controls the height of the barriers
        b=4.0,  # Controls the width of the wells
        shift=0.5,  # shift the well centers
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
        temperature=1.0,
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
        self.temperature = temperature

        # Parameters for the four-well potential
        self.a = a
        self.b = b
        self.shift = shift
        
        # Define the locations of the four wells
        self.well_centers = torch.tensor([
            [-1.5, -1.5],  # Bottom left well
            [-1.5, 1.5],   # Top left well
            [1.5, -1.5],   # Bottom right well
            [1.5, 1.5]     # Top right well
        ], device=device)
        self.well_centers = self.well_centers + self.shift

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
            temperature=temperature,
        )

    def move_to_device(self, device):
        self.well_centers = self.well_centers.to(device)

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

    def force(self, samples, temperature=None):
        samples = samples.requires_grad_(True)
        e = -self.log_prob(samples, temperature=temperature)
        return -torch.autograd.grad(e.sum(), samples)[0]

    def _energy(self, x):
        """Compute the energy of the four-well potential."""
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        energies = torch.zeros(batch_size, device=x.device)
        
        # Calculate the contribution from each well
        for center in self.well_centers:
            # Calculate squared distance to each well center
            dist_sq = torch.sum((x - center)**2, dim=1)
            # Add negative Gaussian contribution (creates wells)
            energies = energies - (self.a * torch.exp(-self.b * dist_sq))
            
        # Add a quadratic confinement to prevent samples from going too far
        confinement = 0.01 * torch.sum(x**2, dim=1)
        
        return energies + confinement

    def sample(self, shape):
        """Sample from the four-well potential using multiple chains."""
        n_samples = shape[0]
        samples_per_well = n_samples // len(self.well_centers)
        remaining_samples = n_samples % len(self.well_centers)
        all_samples = []
        
        # Parameters for MCMC
        n_burnin = 1000
        step_size = 0.05
        
        for well_idx in range(len(self.well_centers)):
            # Determine number of samples for this well
            well_samples = samples_per_well
            if well_idx < remaining_samples:
                well_samples = well_samples + 1
                
            if well_samples == 0:
                continue
                
            # Start near this well
            current = self.well_centers[well_idx] + 0.1 * torch.randn(2, device=self.device)
            current_log_prob = self.log_prob(current)
            
            # Initialize samples for this well
            samples = torch.zeros((well_samples, 2), device=self.device)
            
            # Burn-in phase
            for _ in range(n_burnin):
                proposal = current + step_size * torch.randn(2, device=self.device)
                proposal_log_prob = self.log_prob(proposal)
                
                acceptance_ratio = torch.min(
                    torch.tensor(1.0, device=self.device),
                    torch.exp(proposal_log_prob - current_log_prob)
                )
                
                if torch.rand(1, device=self.device) < acceptance_ratio:
                    current = proposal
                    current_log_prob = proposal_log_prob
            
            # Sampling phase
            for i in range(well_samples):
                proposal = current + step_size * torch.randn(2, device=self.device)
                proposal_log_prob = self.log_prob(proposal)
                
                acceptance_ratio = torch.min(
                    torch.tensor(1.0, device=self.device),
                    torch.exp(proposal_log_prob - current_log_prob)
                )
                
                if torch.rand(1, device=self.device) < acceptance_ratio:
                    current = proposal
                    current_log_prob = proposal_log_prob
                
                samples[i] = current
            
            all_samples.append(samples)
        
        # Combine samples from all wells
        return torch.cat(all_samples, dim=0)

    # @property
    # def log_Z_2D(self):
    #     if self._a == -0.5 and self._b == -6 and self._c == 1.0:
    #         log_Z_dim0 = np.log(11784.50927)
    #         log_Z_dim1 = 0.5 * np.log(2 * torch.pi)
    #         return log_Z_dim0 + log_Z_dim1
    #     else:
    #         raise NotImplementedError

    #####################################################################
    # added for DEM
    #####################################################################

    def get_minima(self):
        """Return the locations of the four minima (well centers)."""
        return self.well_centers.clone()

    def get_true_transition_states(self):
        """Return the saddle points between the wells."""
        # Create saddle points between adjacent wells
        saddle_points = []
        
        # For a four-well system, we have saddle points between:
        # 1. Bottom left and top left
        # 2. Bottom left and bottom right
        # 3. Top left and top right
        # 4. Bottom right and top right
        
        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            # Saddle point is at the midpoint between two adjacent wells
            saddle = (self.well_centers[i] + self.well_centers[j]) / 2
            saddle_points.append(saddle)
            
        return torch.stack(saddle_points)

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


if __name__ == "__main__":
    # Test that rejection sampling is work as desired.
    import matplotlib.pyplot as plt

    # Create the four-well energy function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    four_well = FourWellEnergy(device=device)

    # Visualize the four-well potential
    x = torch.linspace(-3, 3, 100, device=four_well.device)
    y = torch.linspace(-3, 3, 100, device=four_well.device)
    X, Y = torch.meshgrid(x, y)
    Z = four_well.log_prob_2D(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), 50, cmap='viridis')
    plt.colorbar(label='Log Probability')
    plt.title('Four-Well Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
    four_well_samples = four_well.sample((2000,))

    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), 50, cmap='viridis', alpha=0.7)
    plt.scatter(four_well_samples[:, 0].cpu(), four_well_samples[:, 1].cpu(), 
            s=1, color='red', alpha=0.5)
    plt.title('Four-Well Potential with Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
