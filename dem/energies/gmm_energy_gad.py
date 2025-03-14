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
from dem.energies.base_energy_function import BaseGADEnergyFunction
import copy
import traceback


class GMMGADEnergy(GMMEnergy, BaseGADEnergyFunction):

    def __init__(self, *args, **kwargs):
        GMMEnergy.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        BaseGADEnergyFunction.__init__(self, *args, **kwargs)

    #####################################################################
    # GAD
    #####################################################################

    def log_prob(
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
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
        assert (
            samples.shape[-1] == self._dimensionality
        ), "`x` does not match `dimensionality`"

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)

        pseudo_energy, aux_output = self.compute_gad_potential(self._energy, samples)

        if temperature is None:
            temperature = self.temperature
        pseudo_energy = pseudo_energy / temperature

        # convention
        # pseudo_log_prob = pseudo_energy
        pseudo_log_prob = -pseudo_energy

        if return_aux_output:
            return pseudo_log_prob, aux_output
        return pseudo_log_prob

    def physical_potential_log_prob(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of the physical potential (not the GAD potential).
        Same as GMMEnergy.__call__.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Negative of the GMM potential value (scalar)
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        # Compute log-probability of potential energy
        if return_aux_output:
            return self.gmm.log_prob(samples), {}
        return self.gmm.log_prob(samples)

    #####################################################################
    # GAD end
    #####################################################################
