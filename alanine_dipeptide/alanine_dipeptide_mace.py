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

"""Alanine dipeptide with using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""

def rotation_matrix(axis: torch.Tensor, theta: torch.Tensor, device) -> torch.Tensor:
    """
    Return the rotation matrix for a counterclockwise rotation about
    'axis' by angle 'theta' (in radians) using Rodrigues' rotation formula.
    All operations are differentiable.
    """
    # Normalize the axis
    axis = axis / torch.norm(axis)
    
    # Rodrigues rotation formula
    a = torch.cos(theta/2.0)
    b, c, d = -axis * torch.sin(theta/2.0)
    
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    
    R = torch.stack([
        torch.stack([aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)]),
        torch.stack([2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)]),
        torch.stack([2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc])
    ])
    
    return R

def compute_dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Compute the dihedral angle (in radians) defined by four points.
    All operations are differentiable.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1
    b1 = b1 / torch.norm(b1)
    
    # Vector normal to plane 1
    v = b0 - torch.dot(b0, b1) * b1
    
    # Vector normal to plane 2
    w = b2 - torch.dot(b2, b1) * b1
    
    # Angle between normals
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    
    return torch.atan2(y, x)

def set_dihedral(positions: torch.Tensor, indices, target_angle: torch.Tensor) -> torch.Tensor:
    """
    Sets the dihedral angle defined by four atoms to target_angle.
    All operations maintain gradient flow.
    """
    i, j, k, l = indices
    p0, p1, p2, p3 = positions[i], positions[j], positions[k], positions[l]
    
    # Current dihedral angle
    current_angle = compute_dihedral(p0, p1, p2, p3)
    
    # Angle to rotate by
    delta = target_angle - current_angle
    
    # Rotation axis
    axis = positions[k] - positions[j]
    
    # Create rotation matrix (maintains gradients)
    R = rotation_matrix(axis, delta, device=positions.device)
    
    # New positions tensor
    new_positions = positions.clone()
    
    # Origin for rotation
    origin = positions[k]
    
    # Rotate all atoms after l
    rotating_indices = torch.arange(l, positions.shape[0], device=positions.device)
    vectors = positions[rotating_indices] - origin
    
    # Apply rotation (maintains gradients)
    rotated_vectors = torch.matmul(R, vectors.T).T
    new_positions[rotating_indices] = origin + rotated_vectors
    
    return new_positions

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
    
    # Set the dihedral angles.
    # (The following indices are examples. For instance, if the template has the backbone atoms in order,
    #  you might define φ = dihedral(atoms[1,2,3,4]) and ψ = dihedral(atoms[2,3,4,5]).)
    phi_indices = [1, 2, 3, 4]  # <-- Adjust these indices to your template.
    psi_indices = [2, 3, 4, 5]  # <-- Adjust these indices to your template.
    
    positions = set_dihedral(positions, phi_indices, phi)
    positions = set_dihedral(positions, psi_indices, psi)
    
    return positions

def load_alanine_dipeptide_ase(pdbfile):
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
    phi_indices = [1, 2, 3, 4]  # Adjust indices to match template
    psi_indices = [2, 3, 4, 5]  # Adjust indices to match template
    
    positions = set_dihedral(positions, phi_indices, phi_psi[0])
    positions = set_dihedral(positions, psi_indices, phi_psi[1])
    
    # Update positions in ASE Atoms object
    atoms.set_positions(positions.numpy())
    
    return atoms

def update_alanine_dipeptide_with_grad(phi_psi: torch.Tensor, batch: dict) -> dict:
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
    
    # Define dihedral indices
    phi_indices = [4, 6, 8, 14]  # C-N-CA-C for phi
    psi_indices = [6, 8, 14, 16]  # N-CA-C-N for psi
    
    # Apply dihedral rotations (maintaining gradients)
    positions = set_dihedral(positions, phi_indices, phi_psi[0])
    positions = set_dihedral(positions, psi_indices, phi_psi[1])
    
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
    atoms = load_alanine_dipeptide_ase(pdbfile)
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get MACE force field: mace_off or mace_anicc
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

    ################################################
    # placeholder loop for training or plotting
    for i in range(10):
        # Create phi_psi tensor with gradients
        phi_psi = torch.tensor([0.0, 0.0], requires_grad=True)
            
        # Update positions
        batch = update_alanine_dipeptide_with_grad(phi_psi, batch_base)
        
        # TODO: need to update edge_index?
        
        # Test gradient flow
        positions = batch["positions"]
        loss = positions.sum()
        grad = torch.autograd.grad(loss, phi_psi, create_graph=True)[0]
        print("Gradient w.r.t. phi_psi:", grad)
        
        break


if __name__ == "__main__":
    pdbfile = 'alanine_dipeptide/alanine_dipeptide_nowater.pdb'
    
    test_mace_alanine_dipeptide(pdbfile)
    test_mace_alanine_dipeptide_dihedral_grad(pdbfile)




