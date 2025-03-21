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

from alanine_dipeptide.dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_batched,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
)

import torch_geometric as tg

"""Alanine dipeptide with using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""

PDB_FILE = "alanine_dipeptide/data/alanine-dipeptide-nowater.pdb"


def load_alanine_dipeptide_ase():
    """
    Build a full-atom 3D configuration of alanine dipeptide into an ASE Atoms object.

    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Load PDB into ASE
    atoms = ase.io.read(PDB_FILE)
    return fix_atomic_numbers(atoms)


# from mace.calculators.mace import MACECalculator
def repeated_atoms_to_batch(
    calc: mace.calculators.mace.MACECalculator, atoms, bs=1, repeats=1
):
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


def update_alanine_dipeptide_xyz_from_dihedrals_ase(
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


def update_alanine_dipeptide_xyz_from_dihedrals_torch(
    positions: torch.Tensor, phi_psi: torch.Tensor = None, convention="andreas"
):
    """
    Update the positions of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians).

    Parameters:
      phi_psi: Desired φ and ψ dihedral angles (radians)

    Returns:
      positions: torch.Tensor with updated positions
    """
    # Set dihedral angles
    positions1 = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
    positions2 = set_dihedral_torch(positions1, "psi", phi_psi[1], "psi", convention)
    return positions2


def update_alanine_dipeptide_xyz_from_dihedrals_batched(
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

    # if phi_psi is a single value, convert to a "batch" of size 1
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
