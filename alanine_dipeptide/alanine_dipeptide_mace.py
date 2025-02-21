import numpy as np
import torch
import os
import pathlib
import math
from tqdm import tqdm
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

from dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_batched,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
    update_neighborhood_graph_batched,
)
from mace_neighbourshood import get_neighborhood
from alanine_dipeptide_openmm_amber99 import fffile, pdbfile

import torch_geometric as tg

import silence_warnings

"""Alanine dipeptide with using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""


# def build_alanine_dipeptide_openmm(phi, psi, pdbfile):
#     """
#     Build a full-atom 3D configuration of alanine dipeptide with the specified
#     backbone dihedral angles phi and psi (in radians).

#     The function loads a template PDB file and then
#     rotates the appropriate groups of atoms so that the dihedrals match the desired values.

#     Parameters:
#       phi: Desired φ dihedral angle (radians)
#       psi: Desired ψ dihedral angle (radians)

#     Returns:
#       positions_quantity: A simtk.unit.Quantity (shape: [N_atoms, 3]) in nanometers.

#     > **Note:** Adjust the following dihedral atom indices to match your template.
#     """
#     # Load the template structure.
#     pdb = openmm.app.PDBFile(pdbfile)
#     # Extract positions as a NumPy array in nanometers.
#     positions = torch.tensor(pdb.positions.value_in_unit(nanometer))

#     positions = set_dihedral_torch(positions, "phi", phi, "phi")
#     positions = set_dihedral_torch(positions, "psi", psi, "psi")

#     return positions


def load_alanine_dipeptide_ase():
    """
    Build a full-atom 3D configuration of alanine dipeptide using ASE.

    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
      pdbfile: Path to PDB template file

    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Load PDB into ASE
    atoms = ase.io.read(pdbfile)
    return fix_atomic_numbers(atoms)


# from mace.calculators.mace import MACECalculator
def _atoms_to_batch(calc: mace.calculators.mace.MACECalculator, atoms, bs=1, repeats=1):
    config = mace.data.config_from_atoms(atoms, charges_key=calc.charges_key)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace.data.AtomicData.from_config(
                config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.heads
            )
        ]
        * repeats,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(calc.device)
    return batch


def update_alanine_dipeptide_ase(
    atoms, phi_psi: torch.Tensor = None, convention="andreas"
):
    """
    Update the positions of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians) using ASE.

    Parameters:
      phi_psi: Desired φ and ψ dihedral angles (radians)
      pdbfile: Path to PDB template file

    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Get positions as torch tensor
    positions = torch.tensor(atoms.get_positions())

    # Set dihedral angles
    positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
    positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi", convention)

    # Update positions in ASE Atoms object
    atoms.set_positions(positions.numpy())

    return atoms


# DEPRECATED: use update_alanine_dipeptide_with_grad_batched instead
def update_alanine_dipeptide_with_grad(
    phi_psi: torch.Tensor, batch: dict, set_phi=True, set_psi=True, convention="andreas"
) -> dict:
    """
    Update positions based on phi_psi angles while maintaining gradient flow.
    Works with MACE Batch objects.
    """
    # Create a new batch with the same attributes as the input batch
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    new_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    # Get positions tensor and apply dihedral rotations
    positions = batch["positions"].clone()

    if set_phi:
        positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
    if set_psi:
        positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi", convention)

    # Update positions in the new batch
    new_batch["positions"] = positions

    return new_batch


def update_alanine_dipeptide_with_grad_batched(
    phi_psi: torch.Tensor,
    batch: dict,
    set_phi=True,
    set_psi=True,
    convention="andreas",
    return_unbatched=False,
) -> dict:
    """
    Update positions based on phi_psi angles while maintaining gradient flow.
    Works with MACE Batch objects.
    phi_psi is a tensor of shape (B, 2) where each row is [phi, psi], or just (2) for a single configuration.
    """
    # Create a new batch with the same attributes as the input batch
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    new_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    if phi_psi.dim() <= 1:
        phi_psi = phi_psi.unsqueeze(0)

    # Get positions tensor and apply dihedral rotations
    positions = batch["positions"].clone()

    # reshape positions from (B*N_atoms, 3) to (B, N_atoms, 3) based on batch["batch"]
    bs = batch["batch"].max() + 1
    positions = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0)
    positions = torch.stack(list(positions), dim=0).reshape(bs, -1, 3)
    
    if set_phi:
        positions = set_dihedral_torch_batched(
            positions, "phi", phi_psi[:, 0], "phi", convention
        )
    if set_psi:
        positions = set_dihedral_torch_batched(
            positions, "psi", phi_psi[:, 1], "psi", convention
        )
    
    # # REMOVE
    # pos0 = set_dihedral_torch(positions[0].clone().detach(), "phi", phi_psi[0, 0], "phi", convention)
    # pos0 = set_dihedral_torch(pos0, "psi", phi_psi[0, 1], "psi", convention)
    # print(f"diff pos0 and positions[0]: {torch.abs(pos0 - positions[0]).max()}")
    # pos1 = set_dihedral_torch(positions[1].clone().detach(), "phi", phi_psi[1, 0], "phi", convention)
    # pos1 = set_dihedral_torch(pos1, "psi", phi_psi[1, 1], "psi", convention)
    # print(f"diff pos1 and positions[1]: {torch.abs(pos1 - positions[1]).max()}")
    
    # Update positions in the new batch
    if return_unbatched:
        new_batch["positions"] = positions
    else:
        new_batch["positions"] = positions.reshape(-1, 3)

    return new_batch


def fix_atomic_numbers(atoms):
    """
    ASE is misinterpreting the "CA" label from your PDB file.
    atom index 8 is showing up as "Ca" (Calcium, atomic number 20) when it should be "CA" (a Carbon atom).
    In PDB format, "CA" is a special atom name that stands for the alpha Carbon (Cα) of an amino acid, but ASE is interpreting it as the chemical symbol for Calcium.
    Fix atomic numbers in the ASE Atoms object to match what MACE expects.
    For alanine dipeptide, we should only have H(1), C(6), N(7), and O(8).
    """
    atomic_numbers = atoms.get_atomic_numbers()
    # Find any calcium atoms (atomic number 20) and convert them to carbon (atomic number 6)
    atomic_numbers[atomic_numbers == 20] = 6
    atoms.set_atomic_numbers(atomic_numbers)
    return atoms


def compute_energy_and_forces_mace(
    dihedrals: torch.Tensor,
    batch_base: dict = None,
    model: torch.nn.Module = None,
    model_type: str = "off",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the energy and forces for a given set of dihedral angles using the MACE model.
    """
    if batch_base is None or model is None:
        # get alanine dipeptide atoms
        atoms = load_alanine_dipeptide_ase()

        # Get MACE force field: mace_off or mace_anicc
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "off":
            calc = mace_off(model="medium", device=device_str, enable_cueq=True)
        elif model_type == "anicc":
            calc = mace_anicc(device=device_str, enable_cueq=True)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        device = calc.device
        atoms.calc = calc
        atoms_calc = atoms.calc

        ################################################
        # ASE atoms -> torch batch: one time setup
        batch_base = atoms_calc._atoms_to_batch(atoms)

        if atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = atoms_calc._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = atoms_calc.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not atoms_calc.use_compile
        else:
            compute_stress = False
        batch_base = batch_base.to_dict()
        model = atoms_calc.models[0]

    # Create phi_psi tensor with gradients
    phi_psi = dihedrals
    phi_psi.requires_grad = True

    # Update positions
    batch = update_alanine_dipeptide_with_grad(phi_psi, batch_base)

    # need to update edge_index
    # https://github.com/ACEsuit/mace/blob/3e578b02e649a5b2ac8109fa857698fdc42cf842/mace/modules/models.py#L72
    # no gradients for these, but should not affect forces
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=batch["positions"].detach().cpu().numpy(),
        cutoff=model.r_max.item(),
        cell=batch["cell"].detach().cpu().numpy(),
    )
    batch["edge_index"] = torch.tensor(
        edge_index, device=device, dtype=batch["edge_index"].dtype
    )
    batch["shifts"] = torch.tensor(shifts, device=device, dtype=batch["shifts"].dtype)
    batch["unit_shifts"] = torch.tensor(
        unit_shifts, device=device, dtype=batch["unit_shifts"].dtype
    )
    batch["cell"] = torch.tensor(cell, device=device, dtype=batch["cell"].dtype)

    # Compute energy by calling MACE
    out = model(
        batch,
        compute_stress=compute_stress,
        # training=True -> retain_graph when calculating forces=dE/dx
        # which is what we need to compute forces'=dE/dphi_psi
        training=True,  # atoms_calc.use_compile,
    )

    # Compute forces
    forces = torch.autograd.grad(out["energy"], phi_psi, create_graph=True)[0]

    return out["energy"], forces


