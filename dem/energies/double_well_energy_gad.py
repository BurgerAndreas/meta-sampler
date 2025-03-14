import numpy as np
import torch
from typing import Callable
from dem.energies.double_well_energy import DoubleWellEnergy
from dem.energies.base_energy_function import BaseGADEnergyFunction
import os
import sys
import traceback
import copy


class DoubleWellEnergyGAD(DoubleWellEnergy, BaseGADEnergyFunction):
    def __init__(self, *args, **kwargs):
        DoubleWellEnergy.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        BaseGADEnergyFunction.__init__(
            self,
            # gad_offset=gad_offset,
            # clip_energy=clip_energy,
            # stitching=stitching,
            # stop_grad_ev=stop_grad_ev,
            # div_epsilon=div_epsilon,
            # clamp_min=clamp_min,
            # clamp_max=clamp_max,
            *args,
            **kwargs
        )

    def log_prob(self, samples, temperature=None, return_aux_output=False):
        """Compute the log probability of the GAD pseudo-energy of the double well potential.
        E_gad = -E(x) + 1/lambda_1 * (grad(E), v1)^2
        log_prob = -E_gad

        Args:
            samples (torch.Tensor): Samples from the double well potential.
            temperature (float, optional): Temperature. Defaults to None.
            return_aux_output (bool, optional): Whether to return auxiliary output. Defaults to False.

        Returns:
            torch.Tensor: Energy of the double well potential.
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

        # log_prob = -energy
        # but convention of the double well energy is such that we would have to negate it
        # so we don't do anything here
        pseudo_logprob = pseudo_energy

        if return_aux_output:
            return pseudo_logprob, aux_output
        return pseudo_logprob
