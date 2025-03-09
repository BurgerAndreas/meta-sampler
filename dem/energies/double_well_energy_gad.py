import numpy as np
import torch
from typing import Callable
from dem.energies.double_well_energy import DoubleWellEnergy
import os
import sys
import traceback

class DoubleWellEnergyGAD(DoubleWellEnergy):

    def __init__(
        self,
        gad_offset=100.0,
        clip_energy=True,
        stitching=True,
        stop_grad_ev=False,
        div_epsilon=1e-12,
        clamp_min=None,
        clamp_max=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gad_offset = gad_offset
        self.clip_energy = clip_energy
        self.stitching = stitching
        self.stop_grad_ev = stop_grad_ev
        self.div_epsilon = div_epsilon
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, samples, temperature=None, return_aux_output=False):
        return self.log_prob(
            samples, temperature=temperature, return_aux_output=return_aux_output
        )

    def log_prob(self, samples, temperature=None, return_aux_output=False):
        # log_prob = -energy
        if return_aux_output:
            energy, aux_output = self.energy(
                samples, temperature=temperature, return_aux_output=return_aux_output
            )
            return -energy, aux_output
        return -self.energy(samples, temperature=temperature)

    def energy(self, samples, temperature=None, return_aux_output=False):
        """Compute the GAD pseudo-energy of the double well potential.
        E_gad = -E(x) + 1/lambda_1 * (grad(E), v1)^2

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

        #####################################################################
        # Compute energy
        energy = self._energy(samples)

        #####################################################################
        # Compute forces
        def get_energy(x):
            return self._energy(x)

        # Use functorch.grad to compute forces
        try:
            if len(samples.shape) == 1:
                forces = torch.func.grad(get_energy)(samples)
            else:
                forces = torch.vmap(torch.func.grad(get_energy))(samples)
        except Exception as e:
            print(f"Samples: {samples}")
            print(f"Energy: {energy}")
            with open("gad_nan_log.txt", "a") as f:
                f.write(traceback.format_exc())
                f.write(f"Epoch: {self.curr_epoch}\n")
                f.write(f"Samples: {samples}\n")
                f.write(f"Energy: {energy}\n")
                f.write("-" * 80 + "\n")
            raise e

        #####################################################################
        # Compute two smallest eigenvalues of Hessian
        if len(samples.shape) == 1:
            # Handle single sample
            hessian = torch.func.hessian(get_energy)(samples)
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            # Sort eigenvalues and corresponding eigenvectors
            sorted_indices = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            # Get 2 smallest eigenvalues and their eigenvectors
            smallest_eigenvalues = eigenvalues[:2]
            smallest_eigenvectors = eigenvectors[:, :2]
        else:
            # Handle batched inputs using vmap # [B, D, D]
            batched_hessian = torch.vmap(torch.func.hessian(get_energy))(samples)
            # Get eigenvalues and eigenvectors for each sample in batch
            batched_eigenvalues, batched_eigenvectors = torch.linalg.eigh(
                batched_hessian
            )
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
                sorted_indices[..., 0:1]
                .unsqueeze(-1)
                .expand(batched_eigenvectors.shape),
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
                    smallest_eigenvalue < 0, 
                    input=torch.clip(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                        + self.gad_offset,
                        min=self.clamp_min,
                        max=self.clamp_max,
                    ), 
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1]
                )
            else:
                pseudo_energy = torch.where(
                    smallest_eigenvalue < 0, 
                    input=(
                        -energy
                        + (1 / (smallest_eigenvalue + self.div_epsilon))
                        * torch.einsum("bd,bd->b", forces, smallest_eigenvector) ** 2
                    ), 
                    other=-smallest_eigenvalues[..., 0] * smallest_eigenvalues[..., 1]
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
                
        if temperature is None:
            temperature = 1.0
        pseudo_energy = pseudo_energy / temperature

        # convention
        pseudo_energy = -pseudo_energy
        
        if return_aux_output:
            aux_output = {
                "energy": energy,
                "forces": forces,
                "smallest_eigenvalues": smallest_eigenvalues,
                "smallest_eigenvectors": smallest_eigenvectors,
                "pseudo_energy": pseudo_energy,
            }
            return pseudo_energy, aux_output
        return pseudo_energy
