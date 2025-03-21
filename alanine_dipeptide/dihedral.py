import numpy as np
import torch
import math
import os
from typing import Iterable

from openmm.app import PDBFile, ForceField, Simulation, NoCutoff
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity

from alanine_dipeptide.mace_neighbourhood import get_neighborhood

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

# Andreas index invention
# phi_indices = [CB, CA, NL, CLP] = [10, 8, 6, 4]
# psi_indices = [CB, CA, CRP, OR] = [10, 8, 14, 15]
phi_indices_andreas = [10, 8, 6, 4]
psi_indices_andreas = [10, 8, 14, 15]

# Boltzmann Generator. Looks the same as Andreas. Only has a different zero point?
phi_indices_bg = [4, 6, 8, 14]
psi_indices_bg = [6, 8, 14, 16]  # [16, 14, 8, 6]

# phi_indices = phi_indices_andreas
# psi_indices = psi_indices_andreas

phi_indices = phi_indices_bg
psi_indices = psi_indices_bg


def get_indices(indices, convention):
    if indices == "phi":
        if convention == "andreas":
            i, j, k, l = phi_indices_andreas
        elif convention == "bg":
            i, j, k, l = phi_indices_bg
        else:
            raise ValueError(f"Invalid convention: {convention}")
    elif indices == "psi":
        if convention == "andreas":
            i, j, k, l = psi_indices_andreas
        elif convention == "bg":
            i, j, k, l = psi_indices_bg
        else:
            raise ValueError(f"Invalid convention: {convention}")
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


def set_dihedral(
    positions,
    indices,
    target_angle,
    atoms_to_rotate,
    absolute=True,
    convention="andreas",
):
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


def rotation_matrix_torch(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
    return R.to(axis.device)


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
    y = torch.dot(torch.linalg.cross(b1, v), w)
    return torch.atan2(y, x)


def set_dihedral_torch(
    positions: torch.Tensor,
    indices: Iterable[int] | str,
    target_angle: torch.Tensor,
    atoms_to_rotate: Iterable[int] | str,
    convention: str = "andreas",
    absolute: bool = True,
) -> torch.Tensor:
    """
    Adjust the dihedral angle defined by the four atoms specified in 'indices'
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
    target_angle = target_angle.to(positions.device)

    i, j, k, l = get_indices(indices, convention)
    # Use clone to avoid modifying the original tensor
    positions = positions.clone()
    p0 = positions[i].clone()
    p1 = positions[j].clone()
    p2 = positions[k].clone()
    p3 = positions[l].clone()

    current_angle = compute_dihedral_torch(p0, p1, p2, p3)
    if absolute:
        delta = target_angle - current_angle
    else:
        delta = target_angle

    # Define the rotation axis (passing through atoms 1 and 2)
    axis = positions[k] - positions[j]
    axis = axis / torch.norm(axis)

    rotating_indices = get_atoms_to_rotate(atoms_to_rotate)
    origin = positions[k].clone()

    # Compute rotation matrix using torch operations
    R = rotation_matrix_torch(axis, delta)

    # TODO: not vmap-able
    # Rotate the specified atoms.
    for idx in rotating_indices:
        vec = positions[idx] - origin
        rotated_vec = torch.matmul(R, vec)
        positions[idx] = origin + rotated_vec

    return positions


def set_dihedral_torch_vmap(
    positions: torch.Tensor,
    indices: Iterable[int] | str,
    target_angle: torch.Tensor,
    atoms_to_rotate: Iterable[int] | str,
    convention: str = "andreas",
    absolute: bool = True,
) -> torch.Tensor:
    """
    Adjust the dihedral angle defined by the four atoms specified in 'indices'
    to 'target_angle' (in radians) by rotating the part of the molecule specified by atoms_to_rotate.
    This version is built from differentiable torch operations and is compatible with vmap.

    Parameters:
      positions   : Torch tensor of shape (N_atoms, 3)
      indices     : A list or tuple of 4 atom indices [i, j, k, l] defining the dihedral
      target_angle: The desired dihedral angle (in radians), as a torch scalar
      atoms_to_rotate: Either a string ("psi" or "phi") or a list of indices.

    Returns:
      positions   : Modified torch tensor of positions.
    """
    target_angle = target_angle.to(positions.device)

    i, j, k, l = get_indices(indices, convention)
    # Use clone to avoid modifying the original tensor
    positions = positions.clone()
    p0 = positions[i].clone()
    p1 = positions[j].clone()
    p2 = positions[k].clone()
    p3 = positions[l].clone()

    current_angle = compute_dihedral_torch(p0, p1, p2, p3)
    if absolute:
        delta = target_angle - current_angle
    else:
        delta = target_angle

    # Define the rotation axis (passing through atoms 1 and 2)
    axis = positions[k] - positions[j]
    axis = axis / torch.norm(axis)

    rotating_indices = get_atoms_to_rotate(atoms_to_rotate)
    origin = positions[k].clone()

    # Compute rotation matrix using torch operations
    R = rotation_matrix_torch(axis, delta)

    # Create a mask for rotating atoms
    mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    mask[rotating_indices] = True

    # Compute all vectors from origin
    vectors = positions - origin.unsqueeze(0)

    # Apply rotation only to selected atoms
    rotated_vectors = torch.where(
        mask.unsqueeze(1), torch.matmul(R, vectors.unsqueeze(2)).squeeze(2), vectors
    )

    # Update positions out-of-place
    new_positions = origin.unsqueeze(0) + rotated_vectors

    return new_positions


########################################
# Batched Torch version
########################################


def unbatch_alanine_dipeptide(batch: dict) -> dict:
    """
    Unbatch the alanine dipeptide batch.
    """
    bs = batch["batch"].max() + 1
    n_atoms = batch["n_atoms"][0]
    # reshape positions from (B*N_atoms, 3) to (B, N_atoms, 3) based on batch["batch"]
    positions = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0)
    positions = torch.stack(list(positions), dim=0).reshape(bs, n_atoms, 3)
    batch["positions"] = positions
    return batch


def rotation_matrix_torch_batched(
    axis: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """
    Batched Torch version: Return the rotation matrices for a counterclockwise rotation
    about each 'axis' by angle 'theta' (in radians) using Rodrigues’ rotation formula.

    Parameters:
      axis : Tensor of shape (B, 3)
      theta: Tensor of shape (B,)

    Returns:
      R: Tensor of shape (B, 3, 3)
    """
    # Normalize each axis vector.
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    half_theta = theta / 2.0  # shape (B,)
    a = torch.cos(half_theta)  # shape (B,)
    sin_half = torch.sin(half_theta)  # shape (B,)

    # Expand sin_half for broadcasting.
    sin_half = sin_half.unsqueeze(1)  # shape (B, 1)
    v = -axis * sin_half  # shape (B, 3)
    b = v[:, 0]  # shape (B,)
    c = v[:, 1]
    d = v[:, 2]

    aa = a * a  # (B,)
    bb = b * b  # (B,)
    cc = c * c  # (B,)
    dd = d * d  # (B,)
    bc = b * c  # (B,)
    ad = a * d  # (B,)
    ac = a * c  # (B,)
    ab = a * b  # (B,)
    bd = b * d  # (B,)
    cd = c * d  # (B,)

    # Build each row of the rotation matrix.
    row1 = torch.stack(
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], dim=1
    )  # (B, 3)
    row2 = torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], dim=1)
    row3 = torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], dim=1)
    R = torch.stack([row1, row2, row3], dim=1)  # (B, 3, 3)
    return R.to(axis.device)


def compute_dihedral_torch_batched(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> torch.Tensor:
    """
    Batched Torch version: Compute the dihedral angles (in radians) defined by four points
    for each item in the batch.

    Parameters:
      p0, p1, p2, p3: Tensors of shape (B, 3)

    Returns:
      angles: Tensor of shape (B,) containing the dihedral angles.
    """
    b0 = p0 - p1  # (B, 3)
    b1 = p2 - p1  # (B, 3)
    b2 = p3 - p2  # (B, 3)

    b1_norm = torch.norm(b1, dim=1, keepdim=True)  # (B, 1)
    b1 = b1 / b1_norm  # (B, 3)

    # Compute projections to get normals.
    dot_b0_b1 = torch.sum(b0 * b1, dim=1, keepdim=True)  # (B, 1)
    dot_b2_b1 = torch.sum(b2 * b1, dim=1, keepdim=True)  # (B, 1)
    v = b0 - dot_b0_b1 * b1  # (B, 3)
    w = b2 - dot_b2_b1 * b1  # (B, 3)

    x = torch.sum(v * w, dim=1)  # (B,)
    y = torch.sum(torch.cross(b1, v, dim=1) * w, dim=1)  # (B,)
    return torch.atan2(y, x)


def set_dihedral_torch_batched(
    positions: torch.Tensor,
    indices: Iterable[int] | str,
    target_angle: torch.Tensor,
    atoms_to_rotate: Iterable[int] | str,
    convention: str = "andreas",
    absolute: bool = True,
) -> torch.Tensor:
    """
    Batched Torch version: Adjust the dihedral angle defined by the four atoms,
    specified in 'indices', to 'target_angle' (in radians) for a batch of configurations.

    Parameters:
      positions   : Tensor of shape (B, N_atoms, 3) or (N_atoms, 3)
      indices     : A list/tuple of 4 atom indices [i, j, k, l] or a string ("psi"/"phi")
      target_angle: Desired angle (in radians) as a tensor of shape (B,) or a scalar.
      atoms_to_rotate: Either a string ("psi" or "phi") or a list of indices indicating which atoms to rotate.
      convention  : Convention to use for indices.

    Returns:
      positions   : Modified positions tensor of shape (B, N_atoms, 3)
    """
    # if positions is not (N_atoms, 3), then we need to expand it to (B, N_atoms, 3)
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    B = positions.shape[0]

    target_angle = target_angle.to(positions.device)
    if target_angle.dim() == 0:
        # Expand scalar target_angle to all batch items.
        target_angle = target_angle.unsqueeze(0).expand(B)
    elif target_angle.shape[0] != B:
        target_angle = target_angle.expand(B)

    i, j, k, l = get_indices(indices, convention)
    positions = positions.clone()
    p0 = positions[:, i].clone()  # (B, 3)
    p1 = positions[:, j].clone()
    p2 = positions[:, k].clone()
    p3 = positions[:, l].clone()

    current_angle = compute_dihedral_torch_batched(p0, p1, p2, p3)  # (B,)
    if absolute:
        delta = target_angle - current_angle  # (B,)
    else:
        delta = target_angle  # (B,)

    # Define the rotation axis (through atoms j and k).
    axis = positions[:, k] - positions[:, j]  # (B, 3)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)  # (B, 3)

    rotating_indices = get_atoms_to_rotate(atoms_to_rotate)
    origin = positions[:, k].clone()  # (B, 3)

    R = rotation_matrix_torch_batched(axis, delta)  # (B, 3, 3)

    # Rotate the specified atoms.
    for idx in rotating_indices:
        vec = positions[:, idx] - origin  # (B, 3)
        # Batched matrix multiplication.
        rotated_vec = torch.matmul(R, vec.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        positions[:, idx] = origin + rotated_vec

    return positions


#######################################################################################
# Testing
#######################################################################################


def test_set_dihedral_variants():
    """
    Test that the batched variant of set_dihedral_torch produces the same results
    as the unbatched counterpart. Also tests gradient flow through both operations.
    """
    print("-" * 80)
    print("Testing set_dihedral variants...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a test positions tensor simulating a molecule
    N_atoms = 22  # Number of atoms in alanine dipeptide
    positions = torch.randn(N_atoms, 3, device=device, requires_grad=True)

    # Set target angle
    target_angle = torch.tensor(math.pi / 3.0, device=device, requires_grad=True)

    # Test unbatched version
    new_pos_unbatched = set_dihedral_torch(
        positions,
        indices="phi",
        target_angle=target_angle,
        atoms_to_rotate="phi",
        convention="andreas",
    )

    # Test vmap-compatible version
    new_pos_vmap = set_dihedral_torch_vmap(
        positions,
        indices="phi",
        target_angle=target_angle,
        atoms_to_rotate="phi",
        convention="andreas",
    )

    # Test batched version (with batch size 1)
    positions_batch = positions.unsqueeze(0)  # [1, N_atoms, 3]
    new_pos_batched = set_dihedral_torch_batched(
        positions_batch,
        indices="phi",
        target_angle=target_angle,
        atoms_to_rotate="phi",
        convention="andreas",
    ).squeeze(
        0
    )  # Remove batch dimension for comparison

    # Check that all versions produce the same result
    assert torch.allclose(
        new_pos_unbatched, new_pos_vmap, atol=1e-6
    ), "set_dihedral_torch and set_dihedral_torch_vmap produce different results"

    assert torch.allclose(
        new_pos_unbatched, new_pos_batched, atol=1e-6
    ), "set_dihedral_torch and set_dihedral_torch_batched produce different results"

    print("All set_dihedral variants produce the same results!")

    # Test with multiple batch items
    B = 5  # Batch size
    positions_multi_batch = positions.unsqueeze(0).expand(B, -1, -1).clone()
    target_angles = torch.linspace(0, math.pi, B, device=device)

    # Process each item individually
    individual_results = []
    for b in range(B):
        pos_b = set_dihedral_torch(
            positions,
            indices="phi",
            target_angle=target_angles[b],
            atoms_to_rotate="phi",
            convention="andreas",
        )
        individual_results.append(pos_b)
    individual_results = torch.stack(individual_results)

    # Process batch at once
    batched_results = set_dihedral_torch_batched(
        positions_multi_batch,
        indices="phi",
        target_angle=target_angles,
        atoms_to_rotate="phi",
        convention="andreas",
    )

    assert torch.allclose(
        individual_results, batched_results, atol=1e-6
    ), "Batched processing doesn't match individual processing"

    print("Batched processing matches individual processing!")
    print("-" * 80)


def test_batched_variants():
    """
    Test that the batched variants of rotation_matrix_torch, compute_dihedral_torch,
    and set_dihedral_torch produce the same results as their unbatched counterparts.
    Also tests gradient flow through the batched operations.
    """
    print("-" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 5  # Batch size

    # ---- Test rotation_matrix_torch ----
    axis_batch = torch.randn(B, 3, device=device, dtype=torch.float64)
    theta_batch = torch.randn(B, device=device, dtype=torch.float64)
    R_batched = rotation_matrix_torch_batched(axis_batch, theta_batch)  # (B, 3, 3)

    R_individual = []
    for b in range(B):
        R_ind = rotation_matrix_torch(axis_batch[b], theta_batch[b])
        R_individual.append(R_ind)
    R_individual = torch.stack(R_individual, dim=0)
    assert torch.allclose(
        R_batched, R_individual, atol=1e-6
    ), "Batched rotation_matrix_torch does not match individual results."

    # ---- Test compute_dihedral_torch ----
    # Generate random points for dihedral calculation
    p0 = torch.randn(B, 3, device=device, dtype=torch.float64)
    p1 = torch.randn(B, 3, device=device, dtype=torch.float64)
    p2 = torch.randn(B, 3, device=device, dtype=torch.float64)
    p3 = torch.randn(B, 3, device=device, dtype=torch.float64)

    angles_batched = compute_dihedral_torch_batched(p0, p1, p2, p3)
    angles_individual = []
    for b in range(B):
        angle = compute_dihedral_torch(p0[b], p1[b], p2[b], p3[b])
        angles_individual.append(angle)
    angles_individual = torch.stack(angles_individual)
    assert torch.allclose(
        angles_batched, angles_individual, atol=1e-6
    ), "Batched compute_dihedral_torch does not match individual results."

    # ---- Test set_dihedral_torch ----
    # Create a test positions tensor simulating a molecule of N_atoms.
    N_atoms = 22  # (as used in the original code)
    positions_single = torch.randn(N_atoms, 3, device=device, dtype=torch.float64)
    positions_batch = (
        positions_single.unsqueeze(0).expand(B, -1, -1).clone()
    )  # (B, N_atoms, 3)
    target_angle = torch.tensor(math.pi / 3.0, dtype=torch.float64, device=device)

    pos_individual = []
    for b in range(B):
        pos_mod = set_dihedral_torch(
            positions_batch[b].clone(), "psi", target_angle, "psi"
        )
        pos_individual.append(pos_mod)
    pos_individual = torch.stack(pos_individual, dim=0)

    pos_batched = set_dihedral_torch_batched(
        positions_batch.clone(), "psi", target_angle, "psi"
    )
    assert torch.allclose(
        pos_batched, pos_individual, atol=1e-6
    ), "Batched set_dihedral_torch does not match individual results."

    # ---- Test set_dihedral_torch with B=1 ----
    positions_single = torch.randn(N_atoms, 3, device=device, dtype=torch.float64)
    target_angle = torch.tensor(math.pi / 3.0, dtype=torch.float64, device=device)
    pos_batched = set_dihedral_torch_batched(
        positions_single, "psi", target_angle, "psi"
    )
    pos_single = set_dihedral_torch(positions_single, "psi", target_angle, "psi")
    assert torch.allclose(
        pos_batched, pos_single, atol=1e-6
    ), "Batched set_dihedral_torch does not match individual results."

    # ---- Test gradients ----
    # Test gradient flow through batched operations
    # Gradient w.r.t. target angles
    positions_batch.requires_grad_(True)
    target_angles = torch.randn(
        B, device=device, dtype=torch.float64, requires_grad=True
    )
    pos_modified = set_dihedral_torch_batched(
        positions_batch, "psi", target_angles, "psi"
    )
    loss = pos_modified.sum()
    grad_angles = torch.autograd.grad(loss, target_angles, create_graph=True)[0]
    assert (
        grad_angles is not None and not torch.isnan(grad_angles).any()
    ), "Gradient w.r.t. target angles failed"

    # Gradient w.r.t. positions
    positions_batch.requires_grad_(True)
    target_angles = torch.randn(
        B, device=device, dtype=torch.float64, requires_grad=True
    )
    pos_modified = set_dihedral_torch_batched(
        positions_batch, "psi", target_angles, "psi"
    )
    loss = pos_modified.sum()
    grad_pos = torch.autograd.grad(loss, positions_batch, create_graph=True)[0]
    assert (
        grad_pos is not None and not torch.isnan(grad_pos).any()
    ), "Gradient w.r.t. positions failed"

    print("Batched variants test passed: Batched and individual outputs match.")


########################################
# Testing: Compare NumPy vs Torch versions
########################################


def test_numpy_is_torch():
    print("-" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb = PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))
    positions_torch = torch.tensor(
        positions_np, dtype=torch.float64, requires_grad=True, device=device
    )

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
        torch.tensor(target_angle, dtype=torch.float64, device=device),
        "psi",
    )

    # Convert Torch result to NumPy for comparison.
    positions_torch_np = positions_torch_modified.detach().cpu().numpy()

    # Compare the two results.
    assert np.allclose(
        positions_np_modified, positions_torch_np, atol=1e-6
    ), "Mismatch between NumPy and Torch results"
    print("Test passed: NumPy and Torch versions produce the same results.")


def test_set_is_inverse_of_compute():
    print("-" * 80)
    np.random.seed(42)

    # load pdb file
    pdb = PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))

    target_angle = np.pi / 3.0  # 60 degrees

    # test phi
    phi_before = compute_dihedral(
        positions_np[phi_indices_bg[0]],
        positions_np[phi_indices_bg[1]],
        positions_np[phi_indices_bg[2]],
        positions_np[phi_indices_bg[3]],
    )
    print(f"Phi before: {phi_before:.3f} rad")
    positions_np_modified = set_dihedral(
        positions_np.copy(), "phi", target_angle, "phi"
    )
    phi_after = compute_dihedral(
        positions_np_modified[phi_indices_bg[0]],
        positions_np_modified[phi_indices_bg[1]],
        positions_np_modified[phi_indices_bg[2]],
        positions_np_modified[phi_indices_bg[3]],
    )
    print(
        np.allclose(phi_after, target_angle),
        f": set_dihedral and compute_dihedral: {phi_after:.3f} = {target_angle:.3f}",
    )

    # test psi
    psi_before = compute_dihedral(
        positions_np[psi_indices_bg[0]],
        positions_np[psi_indices_bg[1]],
        positions_np[psi_indices_bg[2]],
        positions_np[psi_indices_bg[3]],
    )
    positions_np_modified = set_dihedral(
        positions_np.copy(), "psi", target_angle, "psi"
    )
    psi_after = compute_dihedral(
        positions_np_modified[psi_indices_bg[0]],
        positions_np_modified[psi_indices_bg[1]],
        positions_np_modified[psi_indices_bg[2]],
        positions_np_modified[psi_indices_bg[3]],
    )
    print(
        np.allclose(psi_after, target_angle),
        f": set_dihedral and compute_dihedral: {psi_after:.3f} = {target_angle:.3f}",
    )

    if np.allclose(phi_after, target_angle) and np.allclose(psi_after, target_angle):
        print("Test passed: set_dihedral is inverse of compute_dihedral")
    else:
        print("! Test failed: set_dihedral is not inverse of compute_dihedral")


def test_compute_dihedral_is_mdtraj():
    print("-" * 80)
    np.random.seed(42)

    # load pdb file
    pdb = PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))

    traj = md.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )

    # temporary traj object
    # traj = md.Trajectory(xyz=positions_np, topology=pdb.topology)

    # test phi
    phi_after = compute_dihedral(
        positions_np[phi_indices_bg[0]],
        positions_np[phi_indices_bg[1]],
        positions_np[phi_indices_bg[2]],
        positions_np[phi_indices_bg[3]],
    )
    # compute dihedral with mdtraj
    phi_mdtraj = md.compute_dihedrals(traj, [phi_indices])[0][0]
    print(
        np.allclose(phi_after, phi_mdtraj),
        f": compute_dihedral and mdtraj: {phi_after:.3f} = {phi_mdtraj:.3f}",
    )

    # test psi
    psi_after = compute_dihedral(
        positions_np[psi_indices_bg[0]],
        positions_np[psi_indices_bg[1]],
        positions_np[psi_indices_bg[2]],
        positions_np[psi_indices_bg[3]],
    )
    # compute dihedral with mdtraj
    psi_mdtraj = md.compute_dihedrals(traj, [psi_indices])[0][0]
    print(
        np.allclose(psi_after, psi_mdtraj),
        f": compute_dihedral and mdtraj: {psi_after:.3f} = {psi_mdtraj:.3f}",
    )

    if np.allclose(phi_after, phi_mdtraj) and np.allclose(psi_after, psi_mdtraj):
        print("Test passed: compute_dihedral is equal to mdtraj")
    else:
        print("! Test failed: compute_dihedral is not equal to mdtraj")


def test_gradient_flow():
    # gradient of positions through dihedral angle, to get forces w.r.t. dihedral angle
    print("-" * 80)

    pdb = PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )

    positions = torch.tensor(
        np.array(pdb.positions.value_in_unit(nanometer)),
        dtype=torch.float64,
        requires_grad=True,
    )
    target_angle = torch.tensor(
        math.pi / 3.0, dtype=torch.float64, device=positions.device
    )
    positions_new = set_dihedral_torch(positions, "psi", target_angle, "psi")
    loss = positions_new.sum()
    loss.backward()
    # Print gradient of the original positions.
    print("Gradient with respect to positions:\n", positions.grad)

    # now with gradient w.r.t. dihedral angle
    positions = torch.tensor(
        np.array(pdb.positions.value_in_unit(nanometer)), dtype=torch.float64
    )
    target_angle = torch.tensor(math.pi / 3.0, dtype=torch.float64, requires_grad=True)
    positions_new = set_dihedral_torch(positions, "psi", target_angle, "psi")
    loss = positions_new.sum()
    forces = torch.autograd.grad(loss, target_angle, create_graph=True)[0]
    print("Gradient with respect to dihedral angle:\n", forces)


def test_absolute_vs_relative_rotation():
    # compute_dihedral absolute vs relative rotation
    print("-" * 80)
    # load pdb file
    pdb = PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    positions_np = np.array(pdb.positions.value_in_unit(nanometer))

    target_angle = np.pi / 2.0  # 90 degrees

    # test phi
    phi_before = compute_dihedral(
        positions_np[phi_indices_bg[0]],
        positions_np[phi_indices_bg[1]],
        positions_np[phi_indices_bg[2]],
        positions_np[phi_indices_bg[3]],
    )

    # set absolute angle multiple times
    print("Should be the same:")
    positions_np_modified = positions_np.copy()
    for i in range(5):
        positions_np_modified = set_dihedral(
            positions_np_modified, "phi", target_angle, "phi"
        )
        phi_after = compute_dihedral(
            positions_np_modified[phi_indices_bg[0]],
            positions_np_modified[phi_indices_bg[1]],
            positions_np_modified[phi_indices_bg[2]],
            positions_np_modified[phi_indices_bg[3]],
        )
        # should be the same
        print(f"Phi after {i}: {phi_after:.3f} rad")

    # set relative angle multiple times
    print("-" * 10)
    print("Should be different:")
    positions_np_modified = positions_np.copy()
    for i in range(5):
        positions_np_modified = set_dihedral(
            positions_np_modified, "phi", target_angle, "phi", absolute=False
        )
        phi_after = compute_dihedral(
            positions_np_modified[phi_indices_bg[0]],
            positions_np_modified[phi_indices_bg[1]],
            positions_np_modified[phi_indices_bg[2]],
            positions_np_modified[phi_indices_bg[3]],
        )
        print(f"Phi after {i}: {phi_after:.3f} rad")


if __name__ == "__main__":
    test_numpy_is_torch()
    test_set_dihedral_variants()
    test_absolute_vs_relative_rotation()
    test_set_is_inverse_of_compute()
    test_compute_dihedral_is_mdtraj()
    test_batched_variants()
    test_gradient_flow()
