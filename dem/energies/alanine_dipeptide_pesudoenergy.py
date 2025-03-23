from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image

import copy
import mace
from mace.calculators import mace_off, mace_anicc

# from mace.tools import torch_geometric, torch_tools, utils
import mace.data
import ase
import ase.io
import ase.build

# from ase.calculators.calculator import Calculator, all_changes
# import openmm
# from openmm.unit import nanometer
# from openmm.unit import Quantity

# import torch_geometric as tg
from tqdm import tqdm
import itertools
import pickle

from alanine_dipeptide.mace_neighbourhood import (
    update_neighborhood_graph_batched,
    update_neighborhood_graph_torch,
    update_neighborhood_graph_torch_batched,
    get_neighborhood,
)
from alanine_dipeptide.dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_vmap,
    set_dihedral_torch_batched,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
)
from alanine_dipeptide.alanine_dipeptide_mace import (
    load_alanine_dipeptide_ase,
    repeated_atoms_to_batch,
    update_alanine_dipeptide_xyz_from_dihedrals_batched,
    update_alanine_dipeptide_xyz_from_dihedrals_torch,
)

from dem.energies.base_energy_function import BasePseudoEnergyFunction
from dem.energies.alanine_dipeptide_energy import MaceAlDiEnergy2D, compute_hessians_vmap, tensor_like

# Silence FutureWarning about torch.load weights_only parameter
import warnings
import re

# Create a filter for the specific torch.load FutureWarning
class TorchLoadWarningFilter(warnings.WarningMessage):
    def __init__(self):
        self.pattern = re.compile(r"You are using `torch\.load` with `weights_only=False`")
    
    def __eq__(self, other):
        # Check if this is the torch.load warning we want to filter
        return (isinstance(other, warnings.WarningMessage) and 
                other.category == FutureWarning and 
                self.pattern.search(str(other.message)))

# Register the filter
warnings.filterwarnings(
    "ignore", category=FutureWarning, 
    message="You are using `torch.load` with `weights_only=False`.*"
)



class MaceAlDiPseudoEnergy2D(MaceAlDiEnergy2D, BasePseudoEnergyFunction):
    """
    Pesudoenergy function for the aldanine dipeptide.
    Pseudonergy is defined as the absolute value of the force on the alanine dipeptide.
    Energy and forces are computed using MACE.
    """

    def __init__(self, *args, **kwargs):
        # Initialize base class
        print(f"Initializing MaceAlDiForcePseudoEnergy with kwargs: {kwargs}")
        BasePseudoEnergyFunction.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))
        MaceAlDiEnergy2D.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))

        # transition states of the true energy surface
        self.boundary_points = None
        self.transition_points = None
        self.validation_results = None

    

    def log_prob(
        self,
        samples: torch.Tensor,
        temperature: Optional[float] = None,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of pseudo-energy.

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

        pseudo_energy, aux_output = self._pseudo_potential(samples, return_aux_output=True)

        if temperature is None:
            temperature = self.temperature
        pseudo_energy = pseudo_energy / temperature

        # convention
        # pseudo_log_prob = pseudo_energy
        pseudo_log_prob = -pseudo_energy

        if return_aux_output:
            return pseudo_log_prob, aux_output
        return pseudo_log_prob

    def _pseudo_potential(
        self,
        samples: torch.Tensor,
        return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute pseudo-energy for a batch of samples."""
        if self.use_vmap:
            return self._pseudoenergy_vmap(samples, return_aux_output=return_aux_output)
        else:
            return self._pseudoenergy_batched_loop(samples, return_aux_output=return_aux_output)

    def _pseudoenergy_vmap(
        self,
        samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Takes in samples of phi/psi values and returns the pseudoenergy.
        Args:
            samples (torch.Tensor): [B, 2]
        Returns:
            torch.Tensor: [B]
        """
        minibatch = self.atoms_calc._clone_batch(self.singlebatch_base)
        # minibatch = singlebatch_base.clone()

        def _dihedrals_to_energies(
            _phi_psi, positions, node_attrs, edge_index, batch, head, shifts, ptr
        ):
            # Update xyz positions of atoms based on phi/psi values
            positions1 = set_dihedral_torch_vmap(positions, "phi", _phi_psi[0], "phi", "bg")
            positions2 = set_dihedral_torch_vmap(
                positions1, "psi", _phi_psi[1], "psi", "bg"
            )

            # Update edge indices
            # not vmap-able, because the shape of edge_index is data dependent
            # minibatch = update_neighborhood_graph_torch(
            #     minibatch,
            #     model.r_max.item(),
            # )

            result = self.vectorized_model.forward(
                positions2, node_attrs, edge_index, batch, head, shifts, ptr
            )
            return result["energy"].squeeze(-1)

        # Get input variables as tensors
        positions = minibatch["positions"]
        node_attrs = minibatch["node_attrs"]
        edge_index = minibatch["edge_index"]
        batch = minibatch["batch"]
        head = minibatch["head"][batch] if "head" in minibatch else torch.zeros_like(batch)
        shifts = minibatch["shifts"]
        ptr = minibatch["ptr"]

        # [B]
        _dihedrals_to_energies_vmapped = torch.vmap(
            _dihedrals_to_energies, in_dims=(0, None, None, None, None, None, None, None)
        )
        energies = _dihedrals_to_energies_vmapped(
            samples, positions, node_attrs, edge_index, batch, head, shifts, ptr
        )

        # forces = gradient of energy with respect to phi/psi [B, 2]
        # forces = -torch.func.grad(_dihedrals_to_energies_vmapped, argnums=0)(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)
        forces = -1 * torch.vmap(
            torch.func.grad(_dihedrals_to_energies, argnums=0),
            in_dims=(0, None, None, None, None, None, None, None),
        )(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)


        # compute Hessian [B, 2, 2]
        # hessian = torch.func.hessian(_dihedrals_to_energies_vmapped, argnums=0)(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)
        hessian = torch.vmap(
            torch.func.hessian(_dihedrals_to_energies, argnums=0),
            in_dims=(0, None, None, None, None, None, None, None),
        )(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)

        # [B, D], [B, D, D]
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)


        # Force magnitude [B]
        force_magnitude = torch.linalg.norm(forces, ord=self.forces_norm, dim=-1)
        
        # Hessian eigenvalues and eigenvectors
        # Sort eigenvalues in ascending order and get corresponding indices
        sorted_indices = torch.argsort(eigenvalues, dim=-1)
        # Get sorted eigenvalues
        eigenvalues = torch.gather(
            eigenvalues, -1, sorted_indices
        )
        # Get 2 smallest eigenvalues for each sample
        smallest_eigenvalues = eigenvalues[..., :2]  # [B, 2]
        # Get eigenvectors corresponding to eigenvalues
        smallest_eigenvectors = torch.gather(
            eigenvectors,
            -1,
            sorted_indices[..., 0:1]
            .unsqueeze(-1)
            .expand(eigenvectors.shape),
        )
        eigvalterm = smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1]
        
        # Pseudoenergy [B]
        pseudo_energy = torch.where(
                # smallest_eigenvalue < 0,
                eigvalterm < 0,
                input=torch.clip(
                    force_magnitude
                    # torch.einsum("bd,bd->b", forces, smallest_eigenvectors[..., 0]) ** 2
                    + self.gad_offset,
                    min=self.clamp_min,
                    max=self.clamp_max,
                ),
                other=eigvalterm,
            )

        if return_aux_output:
            aux_output = {}
            return pseudo_energy, aux_output
        return pseudo_energy
    
    def _pseudoenergy_batched_loop(
        self,
        samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """
        Compute pseudoenergy for a batch of dihedral angles
        Args:
            samples (torch.Tensor): [B, 2]
        Returns:
            torch.Tensor: [B]
        """
        # bs = samples.shape[0]

        pseudoenergies = []
        for sample in samples:
            # minibatch = self.singlebatch_base.clone()
            minibatch = self.atoms_calc._clone_batch(minibatch)

            # Update xyz positions of atoms based on phi/psi values
            # forces = gradient of energy with respect to phi/psi
            phi_psi = sample.requires_grad_(True)
            positions = minibatch["positions"]  # [N, 3]
            positions1 = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", "bg")
            minibatch["positions"] = set_dihedral_torch(
                positions1, "psi", phi_psi[1], "psi", "bg"
            )

            # Update edge indices
            edge_index, shifts, unit_shifts, cell = get_neighborhood(
                positions=minibatch["positions"].detach().cpu().numpy(),
                cutoff=self.r_max.item(),
                cell=minibatch["cell"].detach().cpu().numpy(),  # [3, 3]
            )
            minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
            minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
            minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
            minibatch["cell"] = tensor_like(cell, minibatch["cell"])

            # Compute energies
            out = self.model(minibatch, training=True)

            # Compute forces [2]
            forces = torch.autograd.grad(
                outputs=out["energy"],  # [1]
                inputs=phi_psi,  # [2]
                grad_outputs=torch.ones_like(out["energy"], device=self.device),  # [2]
                create_graph=True,
                retain_graph=True,
            )[0]
            forces_norm = torch.linalg.norm(forces)

            hessian = compute_hessians_vmap(forces, phi_psi)  # [d, d]

            # compute smallest eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            smallest_eigenvalues = eigenvalues[:2]
            smallest_eigenvectors = eigenvectors[:, :2]
            eigval_product = smallest_eigenvalues[0] * smallest_eigenvalues[1]

            pseudoenergy = 1.0 * out["energy"] + 1.0 * forces_norm + 1.0 * eigval_product
            pseudoenergies.append(pseudoenergy)

        pseudoenergies = torch.stack(pseudoenergies, dim=0).squeeze(1)
        
        if return_aux_output:
            aux_output = {}
            return pseudoenergies, aux_output
        return pseudoenergies