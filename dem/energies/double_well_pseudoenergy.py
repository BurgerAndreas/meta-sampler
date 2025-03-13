import torch
import numpy as np
import copy
import os
import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot after setting backend

plt.ioff()  # Turn off interactive model

import scipy.optimize
from tqdm import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger

from typing import Optional, Tuple, List, Dict, Any

from dem.energies.base_energy_function import BaseEnergyFunction, BasePseudoEnergyFunction
from dem.energies.double_well_energy import DoubleWellEnergy
from dem.utils.logging_utils import fig_to_image
from dem.utils.plotting import plot_fn, plot_marginal_pair


class DoubleWellPseudoEnergy(DoubleWellEnergy, BasePseudoEnergyFunction):
    """Double well pseudo-energy function to find transition points (index-1 saddle points).
    This function should be minimal at the transition points of some potential energy surface.

    Pseudo-energy that combines potential energy and force terms F=dU/dx,
    and possibly (approximations of) the second order derivatives (Hessian).

    Args:
        dimensionality (int): Dimension of input space
        energy_weight (float): Weight for energy term
        force_weight (float): Weight for force term
        force_exponent_eps (float): If force exponent is negative, add this value to the force magnitude to avoid division by zero. Higher value tends to smear out singularity around |force|=0.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        # Initialize DoubleWellEnergy base class
        print(f"Initializing DoubleWellPseudoEnergy with kwargs: {kwargs}")
        DoubleWellEnergy.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        BasePseudoEnergyFunction.__init__(self, *args, **kwargs)

        self._is_molecule = False

        # transition states of the potential
        self.boundary_points = None
        self.transition_points = None
        self.validation_results = None
    
    def log_prob(
        self, samples: torch.Tensor, temperature: Optional[float] = None, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of DoubleWellPseudoEnergy.
        Corresponds to DoubleWellEnergy.log_prob.

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
            
        pseudo_energy, aux_output = self.compute_pseudo_potential(self._energy, samples)
        
        if temperature is None:
            temperature = self.temperature
        pseudo_energy = pseudo_energy / temperature
        
        # convention
        # pseudo_log_prob = pseudo_energy
        pseudo_log_prob = -pseudo_energy
        
        if return_aux_output:
            return pseudo_log_prob, aux_output
        return pseudo_log_prob