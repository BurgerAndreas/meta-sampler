import numpy as np
import torch
import math

# Import OpenMM modules (using simtk.openmm in OpenMM 7.x; for OpenMM 8+ use openmm)
from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

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
phi_indices = [10, 8, 6, 4]
psi_indices = [10, 8, 14, 15]


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


def set_dihedral(positions, indices, target_angle, atoms_to_rotate):
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

    Returns:
      positions   : Modified NumPy array of positions (in nanometers)
    """
    if indices == "phi":
        i, j, k, l = phi_indices
    elif indices == "psi":
        i, j, k, l = psi_indices
    else:
        i, j, k, l = indices
    p0, p1, p2, p3 = (
        positions[i].copy(),
        positions[j].copy(),
        positions[k].copy(),
        positions[l].copy(),
    )

    target_angle = target_angle % (2 * np.pi)
    current_angle = compute_dihedral(p0, p1, p2, p3)
    delta = target_angle - current_angle

    # Define the rotation axis (passing through atoms j and k)
    axis = positions[k] - positions[j]
    axis /= np.linalg.norm(axis)

    # For simplicity, assume that all atoms with index >= l are to be rotated.
    if atoms_to_rotate == "psi":
        rotating_indices = list(range(l, positions.shape[0]))
    elif atoms_to_rotate == "phi":
        rotating_indices = [0, 1, 2, 3, 4, 5]
    else:
        rotating_indices = atoms_to_rotate
    origin = positions[k].copy()

    R = rotation_matrix(axis, delta)

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


def set_dihedral_torch(positions, indices, target_angle, atoms_to_rotate):
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
    if indices == "phi":
        i, j, k, l = phi_indices
    elif indices == "psi":
        i, j, k, l = psi_indices
    else:
        i, j, k, l = indices

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

    # Determine indices to rotate
    if atoms_to_rotate == "psi":
        rotating_indices = list(range(l, positions.shape[0]))
    elif atoms_to_rotate == "phi":
        rotating_indices = [0, 1, 2, 3, 4, 5]
    else:
        rotating_indices = atoms_to_rotate
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


def test_dihedral_functions():
    # Create a simple set of positions for a molecule (say, 10 atoms).
    np.random.seed(42)
    n_atoms = 10
    # Positions in nanometers (as in OpenMM)
    positions_np = np.random.rand(n_atoms, 3).astype(np.float64)
    # Copy for torch; using double precision for consistency.
    positions_torch = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True
    )

    # Choose a dihedral defined by four indices.
    indices = (1, 2, 3, 4)

    # Set a target dihedral angle (in radians)
    target_angle = math.pi / 3.0  # 60 degrees

    # Decide which atoms to rotate. For this test, assume "psi" rotates indices 4:end.
    atoms_to_rotate = "psi"

    # Run NumPy version.
    positions_np_modified = set_dihedral(
        positions_np.copy(), indices, target_angle, atoms_to_rotate
    )

    # Run Torch version.
    positions_torch_modified = set_dihedral_torch(
        positions_torch,
        indices,
        torch.tensor(target_angle, dtype=torch.float64),
        atoms_to_rotate,
    )

    # Convert Torch result to NumPy for comparison.
    positions_torch_np = positions_torch_modified.detach().numpy()

    # Compare the two results.
    assert np.allclose(
        positions_np_modified, positions_torch_np, atol=1e-6
    ), "Mismatch between NumPy and Torch results"
    print("Test passed: NumPy and Torch versions produce the same results.")


if __name__ == "__main__":
    test_dihedral_functions()

    # Example of computing gradients.
    # Here, we define a simple loss as the sum of positions after dihedral adjustment.
    positions = torch.tensor(
        np.random.rand(10, 3), dtype=torch.float64, requires_grad=True
    )
    indices = (1, 2, 3, 4)
    target_angle = torch.tensor(math.pi / 3.0, dtype=torch.float64)
    atoms_to_rotate = "psi"
    positions_new = set_dihedral_torch(
        positions, indices, target_angle, atoms_to_rotate
    )
    loss = positions_new.sum()
    loss.backward()
    # Print gradient of the original positions.
    print("Gradient with respect to positions:\n", positions.grad)
