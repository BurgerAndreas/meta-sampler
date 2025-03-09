from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot after setting backend

plt.ioff()  # Turn off interactive model

import numpy as np
import torch

# import fab.target_distributions
import fab.target_distributions.gmm
from lightning.pytorch.loggers import WandbLogger

from dem.energies.gmm_energy import GMMEnergy
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image
from dem.utils.plotting import plot_fn, plot_marginal_pair


class GMMGADEnergy(GMMEnergy):

    #####################################################################
    # GAD
    #####################################################################
    def log_prob_energy(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of energy.
        Same as GMMEnergy.__call__.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Negative of pseudo-energy value (scalar)
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        # Compute log-probability of potential energy
        if return_aux_output:
            return self.gmm.log_prob(samples), {}
        return self.gmm.log_prob(samples)

    def gmm_potential(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Alias for log_prob_energy. Same as GMMEnergy.__call__."""
        return self.log_prob_energy(samples, return_aux_output=return_aux_output)

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute pseudo-energy combining energy, force, and Hessian terms.
        Returns unnormalized log-probability = -pseudo-energy.
        Similar to GMMEnergy.__call__.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Negative of pseudo-energy value (scalar)
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.log_prob(samples, return_aux_output=return_aux_output)

    def log_prob(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of GAD pseudo-energy.
        Corresponds to GMMEnergy.log_prob.

        E_GAD = -V(x) + 1/lambda_1 * (grad V(x) dot v_1)^2
        where lambda_1 and v_1 are the smallest eigenvalue and it's corresponding eigenvector of the Hessian of V(x)

        Args:
            samples: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized
            return_aux_output: Whether to return auxiliary outputs

        Returns:
            Negative of pseudo-energy value (scalar)
        """

        #####################################################################
        # Compute energy
        # log_prob = -V(x)
        energy = -self.gmm.log_prob(samples)

        #####################################################################
        # Compute forces
        def get_energy(x):
            # TODO: is sum right here?
            # return self.gmm.log_prob(x).sum()
            return self.gmm.log_prob(x)

        # Use functorch.grad to compute forces
        forces = torch.func.grad(get_energy)(samples)
        # force_magnitude = torch.linalg.norm(forces, ord=self.forces_norm, dim=-1)

        #####################################################################
        # Compute two smallest eigenvalues of Hessian
        if len(samples.shape) == 1:
            # Handle single sample
            hessian = torch.func.hessian(self.gmm.log_prob)(samples)
            eigenvalues = torch.linalg.eigvalsh(hessian)
            smallest_eigenvalues = torch.sort(eigenvalues, dim=-1)[0][:2]
        else:
            # Handle batched inputs using vmap # [B, D, D]
            batched_hessian = torch.vmap(torch.func.hessian(self.gmm.log_prob))(samples)
            # Get eigenvalues for each sample in batch # [B, D]
            batched_eigenvalues = torch.linalg.eigvalsh(batched_hessian)
            # Sort eigenvalues in ascending order for each sample # [B, D]
            batched_eigenvalues = torch.sort(batched_eigenvalues, dim=-1)[0]
            # Get 2 smallest eigenvalues for each sample
            smallest_eigenvalues = batched_eigenvalues[..., :2]  # [B, 2]

        # # Get Hessian-vector product using functorch transforms
        # grad_fn = torch.func.grad(get_energy)
        # def hvp(v):
        #     # Ensure v has the same shape as samples
        #     v = v.reshape(samples.shape)
        #     return torch.func.jvp(grad_fn, (samples,), (v,))[1]

        # # Run Lanczos on the HVP function
        # v0 = torch.randn_like(samples)
        # T, Q_mat = lanczos(hvp, v0, m=100)

        # # Compute eigenvalues of the tridiagonal matrix
        # eigenvalues = torch.linalg.eigvalsh(T)
        # smallest_eigenvalues = eigenvalues[:2]

        #####################################################################
        # Compute GAD potential

        # ensure we have one value per batch
        assert (
            energy.shape == force_magnitude.shape == saddle_bias.shape
        ), f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"
        assert (
            energy.shape[0] == samples.shape[0]
        ), f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"

        # Compute GAD log_prob = - pseudo_energy
        pseudo_energy = (
            -energy
            + 1 / smallest_eigenvalues[0] * (forces * smallest_eigenvalues[1]) ** 2
        )
        pseudo_log_prob = -pseudo_energy

        if return_aux_output:
            aux_output = {
                # "energy": energy,
                # "forces": forces,
                # "hessian": hessian,
                "pseudo_energy": pseudo_energy,
            }
            return pseudo_log_prob, aux_output
        return pseudo_log_prob

    # def __call__(
    #     self, samples: torch.Tensor, return_aux_output: bool = False
    # ) -> torch.Tensor:
    #     """Evaluates GMM log probability at given samples.
    #     Used in train.py and eval.py for computing model loss.

    #     Args:
    #         samples (torch.Tensor): Input points to evaluate

    #     Returns:
    #         torch.Tensor: Log probability values at input points
    #     """
    #     if self.should_unnormalize:
    #         samples = self.unnormalize(samples)

    #     if return_aux_output:
    #         aux_output = {}
    #         return self.gmm.log_prob(samples), aux_output
    #     return self.gmm.log_prob(samples)

    #####################################################################
    # GAD end
    #####################################################################
