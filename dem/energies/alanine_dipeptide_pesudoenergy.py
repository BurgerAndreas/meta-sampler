from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image

import copy
import mace
from mace.calculators import mace_off, mace_anicc
from mace.tools import torch_geometric, torch_tools, utils
import mace.data
import ase
import ase.io
import ase.build
from ase.calculators.calculator import Calculator, all_changes
import openmm
from openmm.unit import nanometer
from openmm.unit import Quantity

import torch_geometric as tg


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
from dem.energies.alanine_dipeptide_energy import MaceAlDiEnergy2D


class MaceAlDiForcePseudoEnergy2D(MaceAlDiEnergy2D, BasePseudoEnergyFunction):
    """
    Pesudoenergy function for the aldanine dipeptide.
    Pseudonergy is defined as the absolute value of the force on the alanine dipeptide.
    Energy and forces are computed using MACE.
    """

    def __init__(self, *args, **kwargs):
        # Initialize base class
        print(f"Initializing MaceAlDiForcePseudoEnergy with kwargs: {kwargs}")
        BasePseudoEnergyFunction.__init__(self, *args, **kwargs)
        MaceAlDiEnergy2D.__init__(self, *copy.deepcopy(args), **copy.deepcopy(kwargs))

        # transition states of the true energy surface
        self.boundary_points = None
        self.transition_points = None
        self.validation_results = None

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the pseudoenergy function at given samples.

        Args:
            samples (torch.Tensor): Input points (dihedral angles)

        Returns:
            torch.Tensor: Energy values at input points
        """
        raise NotImplementedError
        if self.use_vmap:
            return self._forward_vmap(samples)
        else:
            return self._forward_batched(samples)

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
        raise NotImplementedError
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

    def _forward_vmap(self, samples: torch.Tensor) -> torch.Tensor:
        """Takes in a single sample and returns the pseudoenergy.
        No batch dimension
        """
        minibatch = self.minibatch.clone()

        # TODO: does this do anything?
        if self.atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            minibatch = self.atoms_calc._clone_batch(minibatch)
            node_heads = minibatch["head"][minibatch["batch"]]
            num_atoms_arange = torch.arange(minibatch["positions"].shape[0])
            node_e0 = self.model.atomic_energies_fn(minibatch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.atoms_calc.use_compile
        else:
            compute_stress = False

        # Update xyz positions of atoms based on phi/psi values
        # forces = gradient of energy with respect to phi/psi
        phi_psi_batch = samples.requires_grad_(True)
        # TODO: code up non-batched version
        minibatch = update_alanine_dipeptide_xyz_from_dihedrals_batched(
            phi_psi_batch, minibatch, convention=convention
        )

        # Update edge indices
        # TODO: why is the cell so huge?
        minibatch = update_neighborhood_graph_batched(
            minibatch, model.r_max.item(), overwrite_cell=True
        )

        # TODO: non-batched version
        # Compute energies and forces
        out = self.model(minibatch, compute_stress=compute_stress, training=True)
        # [B, 2]
        forces = torch.autograd.grad(
            outputs=out["energy"],  # [B]
            inputs=phi_psi_batch,  # [B, 2]
            grad_outputs=torch.ones_like(out["energy"]),  # [B]
            create_graph=True,
        )[0]

        forces_norm = torch.linalg.norm(forces, dim=1)

        # TODO: normalize energies and maybe forces to a reasonable range
        energy_loss = (self.energy_weight * out["energy"]).mean()
        force_loss = (self.force_weight * forces_norm).mean()
        total_loss = energy_loss + force_loss
        if return_aux_output:
            aux_output = {
                "energy_loss": energy_loss,
                "force_loss": force_loss,
                "total_loss": total_loss,
            }
            return total_loss, aux_output
        return total_loss

    def _forward_batched(self, samples: torch.Tensor) -> torch.Tensor:
        bs = samples.shape[0]
        if bs == self.batch_size:
            minibatch = self.minibatch.clone()
        else:
            # construct a minibatch with the correct batch size
            self.minibatch = repeated_atoms_to_batch(
                self.atoms_calc, copy.deepcopy(self.atoms), bs=bs, repeats=bs
            )

        # TODO: does this do anything?
        if self.atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            minibatch = self.atoms_calc._clone_batch(minibatch)
            node_heads = minibatch["head"][minibatch["batch"]]
            num_atoms_arange = torch.arange(minibatch["positions"].shape[0])
            node_e0 = self.model.atomic_energies_fn(minibatch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.atoms_calc.use_compile
        else:
            compute_stress = False

        # Update xyz positions of atoms based on phi/psi values
        # forces = gradient of energy with respect to phi/psi
        phi_psi_batch = samples.requires_grad_(True)
        minibatch = update_alanine_dipeptide_xyz_from_dihedrals_batched(
            phi_psi_batch, minibatch, convention=convention
        )

        # Update edge indices
        # TODO: why is the cell so huge?
        minibatch = update_neighborhood_graph_batched(
            minibatch, model.r_max.item(), overwrite_cell=True
        )

        # Compute energies and forces for all configurations
        out = self.model(minibatch, compute_stress=compute_stress, training=True)
        # [B, 2]
        forces = torch.autograd.grad(
            outputs=out["energy"],  # [B]
            inputs=phi_psi_batch,  # [B, 2]
            grad_outputs=torch.ones_like(out["energy"]),  # [B]
            create_graph=True,
        )[0]

        forces_norm = torch.linalg.norm(forces, dim=1)

        # TODO: normalize energies and maybe forces to a reasonable range
        pseudoenergy = (
            self.energy_weight * out["energy"] + self.force_weight * forces_norm
        )
        return pseudoenergy
