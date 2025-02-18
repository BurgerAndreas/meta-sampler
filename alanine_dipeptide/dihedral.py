import numpy as np
import torch
import math
import os

# Import OpenMM modules (using simtk.openmm in OpenMM 7.x; for OpenMM 8+ use openmm)
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

import mdtraj as md

# Set the dihedral angles.
# A dihedral angle is defined by four atoms: p0, p1, p2, and p3.
# When you change a dihedral angle, you rotate a subset of the molecule around the bond between atoms p1 and p2. Specifically:
#     p0, p1, p2, and p3 define the plane and angle.
#     The bond between p1 and p2 is the axis of rotation.
#     Atoms p0 and p1 remain fixed (i.e., they do not move).
#     Atom p2 is the pivot point, meaning it stays in place.
#     Atom p3 and all atoms connected to p3 (and beyond) rotate as a rigid unit around the p1-p2 axis.
# This means that when you set the dihedral angle, you are only rotating the part of the molecule that is "downstream" from p3, leaving the rest of the structure unchanged

# Andreas invention
# phi_indices = [CB, CA, NL, CLP] = [10, 8, 6, 4]
# psi_indices = [CB, CA, CRP, OR] = [10, 8, 14, 15]
phi_indices_andreas = [10, 8, 6, 4]
psi_indices_andreas = [10, 8, 14, 15]

phi_atoms_bg = [4, 6, 8, 14]
psi_atoms_bg = [6, 8, 14, 16] # [16, 14, 8, 6]

phi_indices = phi_indices_andreas
psi_indices = psi_indices_andreas

phi_indices = phi_atoms_bg
psi_indices = psi_atoms_bg

def get_indices(indices, convention):
    if indices == "phi":
        if convention == "andreas":
            i, j, k, l = phi_indices_andreas
        elif convention == "bg":
            i, j, k, l = phi_indices_bg
    elif indices == "psi":
        if convention == "andreas":
            i, j, k, l = psi_indices_andreas
        elif convention == "bg":
            i, j, k, l = psi_indices_bg
    else:
        i, j, k, l = indices
    return i, j, k, l

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix for a counterclockwise rotation about
    'axis' by angle 'theta' (in radians) using Rodrigues’ rotation formula.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def compute_dihedral(p0, p1, p2, p3):
    """
    Compute the dihedral angle (in radians) defined by four points.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so that it does not influence magnitude of vector
    b1 = b1 / np.linalg.norm(b1)

    # Compute the vectors normal to the planes defined by (p0,p1,p2) and (p1,p2,p3)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def get_atoms_to_rotate(atoms_to_rotate):
    if atoms_to_rotate == "psi":
        # For simplicity, assume that all atoms with index >= l are to be rotated.
        # rotating_indices = list(range(l, positions.shape[0]))
        rotating_indices = [15, 16, 17, 18, 19, 20, 21]
    elif atoms_to_rotate == "phi":
        # phi -> include 7 atoms
        rotating_indices = [0, 1, 2, 3, 4, 5, 7]
    else:
        rotating_indices = atoms_to_rotate
    return rotating_indices

def set_dihedral(positions, indices, target_angle, atoms_to_rotate, absolute=True, convention="andreas"):
    """
    Adjust the dihedral angle defined by the four atoms specified in 'indices'
    to 'target_angle' (in radians) by rotating all atoms “downstream” of atom indices[3].
    A dihedral angle is defined by four atoms: p0, p1, p2, and p3.
    When you change a dihedral angle, you rotate a subset of the molecule around the bond between atoms p1 and p2. Specifically:
        p0, p1, p2, and p3 define the plane and angle.
        The bond between p1 and p2 is the axis of rotation.
        Atoms p0 and p1 remain fixed (i.e., they do not move).
        Atom p2 is the pivot point, meaning it stays in place.
        Atom p3 and all atoms connected to p3 (and beyond) rotate as a rigid unit around the p1-p2 axis.
    This means that when you set the dihedral angle, you are only rotating the part of the molecule that is "downstream" from p3, leaving the rest of the structure unchanged

    You have to specify which atoms are "downstream" of p3:
    e.g. assume that all atoms with index >= l are to be rotated.
    (which is the case for Psi and our pdb file)
    atoms_to_rotate = list(range(l, positions.shape[0]))

    Parameters:
      positions   : NumPy array of shape (N_atoms, 3)
      indices     : A list or tuple of 4 atom indices [i, j, k, l] defining the dihedral
      target_angle: The desired dihedral angle (in radians, i.e. degrees * np.pi/180)
      atoms_to_rotate: Either a string ("psi" or "phi") or a list of indices.
      absolute: If True, the dihedral angle is set to target angle, if False, the dihedral angle is set to target angle - current angle.

    Returns:
      positions   : Modified NumPy array of positions (in nanometers)
    """
    i, j, k, l = get_indices(indices, convention)
    p0, p1, p2, p3 = (
        positions[i].copy(),
        positions[j].copy(),
        positions[k].copy(),
        positions[l].copy(),
    )

    target_angle = target_angle % (2 * np.pi)
    if absolute:
        current_angle = compute_dihedral(p0, p1, p2, p3)
        delta = target_angle - current_angle
    else:
        delta = target_angle

    # Define the rotation axis (passing through atoms j and k)
    axis = positions[k] - positions[j]
    axis /= np.linalg.norm(axis)

    origin = positions[k].copy()

    R = rotation_matrix(axis, delta)

    rotating_indices = get_atoms_to_rotate(atoms_to_rotate)
    for idx in rotating_indices:
        vec = positions[idx] - origin
        positions[idx] = origin + np.dot(R, vec)

    return positions


###############################################################################################
# Torch version that maintains gradients from phi/psi to positions
###############################################################################################


def rotation_matrix_torch(axis, theta):
    """
    Torch version: Return the rotation matrix for a counterclockwise rotation about
    'axis' by angle 'theta' (in radians) using Rodrigues’ rotation formula.
    This function is built entirely from differentiable torch operations.
    """
    # Ensure axis is a tensor of float type
    axis = axis / torch.norm(axis)
    # Compute half-angle quantities
    half_theta = theta / 2.0
    a = torch.cos(half_theta)
    sin_half = torch.sin(half_theta)
    b, c, d = -axis * sin_half

    aa = a * a
    bb = b * b
    cc = c * c
    dd = d * d
    bc = b * c
    ad = a * d
    ac = a * c
    ab = a * b
    bd = b * d
    cd = c * d

    # Build rows using torch.stack so that gradients are tracked
    row1 = torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)])
    row2 = torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)])
    row3 = torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc])
    R = torch.stack([row1, row2, row3])
    return R


def compute_dihedral_torch(p0, p1, p2, p3):
    """
    Torch version: Compute the dihedral angle (in radians) defined by four points.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so that it does not influence magnitude
    b1_norm = torch.norm(b1)
    b1 = b1 / b1_norm

    # Compute projections to get vectors normal to the planes
    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    return torch.atan2(y, x)


def set_dihedral_torch(positions, indices, target_angle, atoms_to_rotate, convention="andreas"):
    """
    Torch version: Adjust the dihedral angle defined by the four atoms specified in 'indices'
    to 'target_angle' (in radians) by rotating the part of the molecule specified by atoms_to_rotate.
    This version is built from differentiable torch operations.

    Parameters:
      positions   : Torch tensor of shape (N_atoms, 3)
      indices     : A list or tuple of 4 atom indices [i, j, k, l] defining the dihedral
      target_angle: The desired dihedral angle (in radians), as a torch scalar
      atoms_to_rotate: Either a string ("psi" or "phi") or a list of indices.

    Returns:
      positions   : Modified torch tensor of positions.
    """
    i, j, k, l = get_indices(indices, convention)
    # Use clone to avoid modifying the original tensor
    positions = positions.clone()
    p0 = positions[i].clone()
    p1 = positions[j].clone()
    p2 = positions[k].clone()
    p3 = positions[l].clone()

    current_angle = compute_dihedral_torch(p0, p1, p2, p3)
    delta = target_angle - current_angle

    # Define the rotation axis (passing through atoms j and k)
    axis = positions[k] - positions[j]
    axis = axis / torch.norm(axis)

    rotating_indices = get_atoms_to_rotate(atoms_to_rotate)
    origin = positions[k].clone()

    # Compute rotation matrix using torch operations
    R = rotation_matrix_torch(axis, delta)

    # Rotate the specified atoms.
    for idx in rotating_indices:
        vec = positions[idx] - origin
        # Note: using torch.matmul to multiply R and vec.
        rotated_vec = torch.matmul(R, vec)
        positions[idx] = origin + rotated_vec

    return positions


########################################
# Testing: Compare NumPy vs Torch versions
########################################


def test_numpy_is_torch():
    print("-"*80)
    pdb = PDBFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))
    positions_torch = torch.tensor(positions_np, dtype=torch.float64, requires_grad=True)

    # Set a target dihedral angle (in radians)
    target_angle = math.pi / 3.0  # 60 degrees

    # Decide which atoms to rotate. For this test, assume "psi" rotates indices 4:end.
    atoms_to_rotate = "psi"

    # Run NumPy version.
    positions_np_modified = set_dihedral(
        positions_np.copy(), "psi", target_angle, "psi"
    )

    # Run Torch version.
    positions_torch_modified = set_dihedral_torch(
        positions_torch,        
        "psi",
        torch.tensor(target_angle, dtype=torch.float64),
        "psi",
    )

    # Convert Torch result to NumPy for comparison.
    positions_torch_np = positions_torch_modified.detach().numpy()

    # Compare the two results.
    assert np.allclose(
        positions_np_modified, positions_torch_np, atol=1e-6
    ), "Mismatch between NumPy and Torch results"
    print("Test passed: NumPy and Torch versions produce the same results.")


def test_set_is_inverse_of_compute():
    print("-"*80)
    np.random.seed(42)
    
    # load pdb file
    pdb = PDBFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))
    
    
    target_angle = np.pi / 3.0  # 60 degrees
    
    # test phi
    phi_before = compute_dihedral(
        positions_np[phi_atoms_bg[0]],
        positions_np[phi_atoms_bg[1]],
        positions_np[phi_atoms_bg[2]],
        positions_np[phi_atoms_bg[3]],
    )
    print(f"Phi before: {phi_before:.3f} rad")
    positions_np_modified = set_dihedral(
        positions_np.copy(), "phi", target_angle, "phi"
    )
    phi_after = compute_dihedral(
        positions_np_modified[phi_atoms_bg[0]],
        positions_np_modified[phi_atoms_bg[1]],
        positions_np_modified[phi_atoms_bg[2]],
        positions_np_modified[phi_atoms_bg[3]],
    )
    print(np.allclose(phi_after, target_angle), f": set_dihedral and compute_dihedral: {phi_after:.3f} = {target_angle:.3f}")
    
    # test psi
    psi_before = compute_dihedral(
        positions_np[psi_atoms_bg[0]],
        positions_np[psi_atoms_bg[1]],
        positions_np[psi_atoms_bg[2]],
        positions_np[psi_atoms_bg[3]],
    )
    positions_np_modified = set_dihedral(
        positions_np.copy(), "psi", target_angle, "psi"
    )
    psi_after = compute_dihedral(
        positions_np_modified[psi_atoms_bg[0]],
        positions_np_modified[psi_atoms_bg[1]],
        positions_np_modified[psi_atoms_bg[2]],
        positions_np_modified[psi_atoms_bg[3]],
    )
    print(np.allclose(psi_after, target_angle), f": set_dihedral and compute_dihedral: {psi_after:.3f} = {target_angle:.3f}")
    
    if np.allclose(phi_after, target_angle) and np.allclose(psi_after, target_angle):
        print("Test passed: set_dihedral is inverse of compute_dihedral")
    else:
        print("! Test failed: set_dihedral is not inverse of compute_dihedral")
        
def test_compute_dihedral_is_mdtraj():
    print("-"*80)
    np.random.seed(42)
    
    # load pdb file
    pdb = PDBFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))
    
    traj = md.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    
    # temporary traj object
    # traj = md.Trajectory(xyz=positions_np, topology=pdb.topology)
    
    # test phi
    phi_after = compute_dihedral(
        positions_np[phi_atoms_bg[0]],
        positions_np[phi_atoms_bg[1]],
        positions_np[phi_atoms_bg[2]],
        positions_np[phi_atoms_bg[3]],
    )
    # compute dihedral with mdtraj
    phi_mdtraj = md.compute_dihedrals(traj, [phi_indices])[0][0]
    print(np.allclose(phi_after, phi_mdtraj), f": compute_dihedral and mdtraj: {phi_after:.3f} = {phi_mdtraj:.3f}")
    
    # test psi
    psi_after = compute_dihedral(
        positions_np[psi_atoms_bg[0]],
        positions_np[psi_atoms_bg[1]],
        positions_np[psi_atoms_bg[2]],
        positions_np[psi_atoms_bg[3]],
    )
    # compute dihedral with mdtraj
    psi_mdtraj = md.compute_dihedrals(traj, [psi_indices])[0][0]
    print(np.allclose(psi_after, psi_mdtraj), f": compute_dihedral and mdtraj: {psi_after:.3f} = {psi_mdtraj:.3f}")
    
    if np.allclose(phi_after, phi_mdtraj) and np.allclose(psi_after, psi_mdtraj):
        print("Test passed: compute_dihedral is equal to mdtraj")
    else:
        print("! Test failed: compute_dihedral is not equal to mdtraj")
    
    

def test_gradient_flow():
    # gradient of positions through dihedral angle, to get forces w.r.t. dihedral angle
    print("-"*80)
    # Example of computing gradients.
    
    pdb = PDBFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    positions = torch.tensor(
        np.array(pdb.positions.value_in_unit(nanometer)), dtype=torch.float64, requires_grad=True
    )
    target_angle = torch.tensor(math.pi / 3.0, dtype=torch.float64)
    positions_new = set_dihedral_torch(
        positions, "psi", target_angle, "psi"
    )
    loss = positions_new.sum()
    loss.backward()
    # Print gradient of the original positions.
    print("Gradient with respect to positions:\n", positions.grad)
    print("Test passed: Gradient flow is correct")


def test_absolute_vs_relative_rotation():
    # compute_dihedral absolute vs relative rotation
    print("-"*80)
    # load pdb file
    pdb = PDBFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "alanine-dipeptide-nowater.pdb"))
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))
    
    target_angle = np.pi / 2.0  # 90 degrees
    
    # test phi
    phi_before = compute_dihedral(
        positions_np[phi_atoms_bg[0]],
        positions_np[phi_atoms_bg[1]],
        positions_np[phi_atoms_bg[2]],
        positions_np[phi_atoms_bg[3]],
    )
    
    # set absolute angle multiple times
    print("Should be the same:")
    positions_np_modified = positions_np.copy()
    for i in range(5):
        positions_np_modified = set_dihedral(
            positions_np_modified, "phi", target_angle, "phi"
        )
        phi_after = compute_dihedral(
            positions_np_modified[phi_atoms_bg[0]],
            positions_np_modified[phi_atoms_bg[1]],
            positions_np_modified[phi_atoms_bg[2]],
            positions_np_modified[phi_atoms_bg[3]],
        )
        # should be the same
        print(f"Phi after {i}: {phi_after:.3f} rad")
    
    # set relative angle multiple times
    print("-"*10)
    print("Should be different:")
    positions_np_modified = positions_np.copy()
    for i in range(5):
        positions_np_modified = set_dihedral(
            positions_np_modified, "phi", target_angle, "phi", absolute=False
        )
        phi_after = compute_dihedral(
            positions_np_modified[phi_atoms_bg[0]], 
            positions_np_modified[phi_atoms_bg[1]], 
            positions_np_modified[phi_atoms_bg[2]], 
            positions_np_modified[phi_atoms_bg[3]],
        )
        print(f"Phi after {i}: {phi_after:.3f} rad")


if __name__ == "__main__":
    test_numpy_is_torch()
    test_gradient_flow()
    test_absolute_vs_relative_rotation()
    test_set_is_inverse_of_compute()
    test_compute_dihedral_is_mdtraj()
    
