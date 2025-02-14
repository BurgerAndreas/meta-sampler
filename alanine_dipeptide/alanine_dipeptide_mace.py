import numpy as np
import torch
import os
import pathlib

import mace
from mace.calculators import mace_off, mace_anicc
import ase
import ase.io
import ase.build
from ase.calculators.calculator import Calculator, all_changes
import openmm
from openmm.unit import nanometer
from openmm.unit import Quantity

from dihedral import set_dihedral_torch

"""Alanine dipeptide with using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""


def build_alanine_dipeptide_openmm(phi, psi, pdbfile):
    """
    Build a full-atom 3D configuration of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians).
    
    The function loads a template PDB file and then
    rotates the appropriate groups of atoms so that the dihedrals match the desired values.
    
    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
    
    Returns:
      positions_quantity: A simtk.unit.Quantity (shape: [N_atoms, 3]) in nanometers.
    
    > **Note:** Adjust the following dihedral atom indices to match your template.
    """
    # Load the template structure.
    pdb = openmm.app.PDBFile(pdbfile)
    # Extract positions as a NumPy array in nanometers.
    positions = torch.tensor(pdb.positions.value_in_unit(nanometer))
    
    positions = set_dihedral_torch(positions, "phi", phi, "phi")
    positions = set_dihedral_torch(positions, "psi", psi, "psi")
    
    return positions

def load_alanine_dipeptide_ase(pdbfile=None):
    """
    Build a full-atom 3D configuration of alanine dipeptide using ASE.
    
    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
      pdbfile: Path to PDB template file
    
    Returns:
      atoms: ASE Atoms object with updated positions
    """
    if pdbfile is None:
        pdbfile = 'alanine_dipeptide/data/alanine_dipeptide_nowater.pdb'
    # Load PDB into ASE
    atoms = ase.io.read(pdbfile)
    return fix_atomic_numbers(atoms)

def update_alanine_dipeptide_ase(atoms, phi_psi: torch.Tensor = None):
    """
    Update the positions of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians) using ASE.
    
    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
      pdbfile: Path to PDB template file
    
    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Get positions as torch tensor
    positions = torch.tensor(atoms.get_positions())
    
    # Set dihedral angles    
    positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi")
    positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi")
    
    # Update positions in ASE Atoms object
    atoms.set_positions(positions.numpy())
    
    return atoms

def update_alanine_dipeptide_with_grad(phi_psi: torch.Tensor, batch: dict, set_phi=True, set_psi=True) -> dict:
    """
    Update positions based on phi_psi angles while maintaining gradient flow.
    Works with MACE Batch objects.
    """
    # Create a new batch with the same attributes as the input batch
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    new_batch = {key: value.clone() if torch.is_tensor(value) else value 
                 for key, value in batch.items()}
    
    # Get positions tensor and apply dihedral rotations
    positions = batch["positions"].clone()
    
    if set_phi:
        positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi")
    if set_psi:
        positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi")
    
    # Update positions in the new batch
    new_batch["positions"] = positions
    
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

def compute_energy_and_forces_mace(dihedrals: torch.Tensor, pdbfile: str, batch_base: dict = None, model: torch.nn.Module = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the energy and forces for a given set of dihedral angles using the MACE model.
    """
    if batch_base is None or model is None:
        # get alanine dipeptide atoms
        atoms = load_alanine_dipeptide_ase(pdbfile)
            
        # Get MACE force field: mace_off or mace_anicc
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        calc = mace_off(model="medium", device=device_str) # enable_cueq=True
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
    
    # TODO: need to update edge_index?
    
    # Compute energy by calling MACE
    out = model(
        batch,
        compute_stress=compute_stress,
        # training=True -> retain_graph when calculating forces=dE/dx
        # which is what we need to compute forces'=dE/dphi_psi
        training=True, #atoms_calc.use_compile,
    )
    
    # Compute forces
    forces = torch.autograd.grad(out["energy"], phi_psi, create_graph=True)[0]
    
    return out["energy"], forces


def test_mace_alanine_dipeptide(pdbfile):
    print("-"*80)
    atoms = load_alanine_dipeptide_ase(pdbfile)
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print out the atoms list
    print("\nList of atoms:")
    print("Index  Symbol  Atomic#  Position (x, y, z)")
    print("-" * 45)
    for i, (sym, num, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_atomic_numbers(), atoms.get_positions())):
        print(f"{i:3d}     {sym:2s}      {num:2d}      ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")

    # Get MACE force field
    calc = mace_off(model="medium", device=device_str) # enable_cueq=True
    device = calc.device
    atoms.calc = calc

    # Test energy
    print("Energy:", atoms.get_potential_energy())

    # Update positions
    atoms = update_alanine_dipeptide_ase(atoms, phi_psi=torch.tensor([0.0, 0.0]))
    print("Energy:", atoms.get_potential_energy())
    # print("Forces:", atoms.get_forces())
    return True
    

def test_mace_alanine_dipeptide_dihedral_grad(pdbfile):
    # compute force w.r.t. phi and psi
    print("-"*80)
    
    # get alanine dipeptide atoms
    atoms = load_alanine_dipeptide_ase(pdbfile)
    
    # Get MACE force field: mace_off or mace_anicc
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc = mace_off(model="medium", device=device_str) # enable_cueq=True
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
    
    # save default configuration
    ase.io.write(f"alanine_dipeptide/outputs/default.xyz", atoms)
    
    # phi_indices = [6, 8]  # C-N-CA-C for phi
    # psi_indices = [6, 8, 14, 16]  # N-CA-C-N for psi
    phi_indices = [1, 2, 3, 4]  # Adjust indices to match template
    psi_indices = [2, 3, 4, 5]
    
    # only plot atoms in phi_indices
    _atoms = atoms.copy()
    for _a in _atoms:
        if _a.index not in phi_indices:
            _a.symbol = 'He'
    ase.io.write(f"alanine_dipeptide/outputs/default_phi_atoms.xyz", _atoms)
    # only plot atoms in psi_indices
    _atoms = atoms.copy()
    for _a in _atoms:
        if _a.index not in psi_indices:
            _a.symbol = 'He'
    ase.io.write(f"alanine_dipeptide/outputs/default_psi_atoms.xyz", _atoms)
    
    # set phi and psi to 0
    batch = update_alanine_dipeptide_with_grad([0.0, 0.0], batch_base, set_phi=True, set_psi=False)
    _atoms = atoms.copy()
    _atoms.set_positions(batch["positions"].detach().cpu().numpy())
    ase.io.write(f"alanine_dipeptide/outputs/default_phi0.xyz", _atoms)
    batch = update_alanine_dipeptide_with_grad([0.0, 0.0], batch_base, set_phi=False, set_psi=True)
    _atoms = atoms.copy()
    _atoms.set_positions(batch["positions"].detach().cpu().numpy())
    ase.io.write(f"alanine_dipeptide/outputs/default_psi0.xyz", _atoms)

    ################################################
    # placeholder loop for training or plotting
    angles = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    for i, phi_psi in enumerate(angles):
        # Create phi_psi tensor with gradients
        phi_psi = torch.tensor(phi_psi, requires_grad=True)
            
        # Update positions
        batch = update_alanine_dipeptide_with_grad(phi_psi, batch_base)
        
        # TODO: need to update edge_index?
        
        # Test gradient flow
        # testgrad = torch.autograd.grad(batch["positions"].sum(), phi_psi, create_graph=True)[0]
        # print("Test gradient w.r.t. phi_psi:", testgrad)
        
        # Compute energy by calling MACE
        out = model(
            batch,
            compute_stress=compute_stress,
            # training=True -> retain_graph when calculating forces=dE/dx
            # which is what we need to compute forces'=dE/dphi_psi
            training=True, #atoms_calc.use_compile,
        )
        
        # Compute forces
        forces = torch.autograd.grad(out["energy"], phi_psi, create_graph=True)[0]
        print(f"Forces w.r.t. phi_psi: {forces}")
        
        # save as .xyz file
        atoms.set_positions(batch["positions"].detach().cpu().numpy())
        ase.io.write(f"alanine_dipeptide/outputs/phi{phi_psi[0]:.1f}_psi{phi_psi[1]:.1f}.xyz", atoms)


if __name__ == "__main__":
    pdbfile = 'alanine_dipeptide/data/alanine_dipeptide_nowater.pdb'
    
    test_mace_alanine_dipeptide(pdbfile)
    test_mace_alanine_dipeptide_dihedral_grad(pdbfile)




