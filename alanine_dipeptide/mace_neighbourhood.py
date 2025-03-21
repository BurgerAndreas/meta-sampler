# from
# https://github.com/ACEsuit/mace/blob/3e578b02e649a5b2ac8109fa857698fdc42cf842/mace/data/neighborhood.py

from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list
import torch


def tensor_like(_new, _base):
    return torch.tensor(_new, device=_base.device, dtype=_base.dtype)


# from openmm / bgflow
def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a neighborhood graph for atomic systems, determining which atoms are within the cutoff distance of each other.

    Args:
        positions: NumPy array of shape [num_positions, 3] containing atomic coordinates
        cutoff: Float value defining the maximum distance for considering atoms as neighbors
        pbc: Optional tuple of 3 booleans indicating periodic boundary conditions in x, y, z directions
        cell: Optional 3×3 NumPy array defining the unit cell vectors
        true_self_interaction: Boolean flag to control whether self-interactions are included

    Returns:
        edge_index: NumPy array of shape [2, n_edges] containing indices of connected atoms (sender, receiver)
        shifts: NumPy array of shape [n_edges, 3] containing the shift vectors for each edge in Cartesian coordinates.
               These represent the actual displacement vectors needed to move from one atom to its neighbor when
               considering periodic boundary conditions.
        unit_shifts: NumPy array of shape [n_edges, 3] containing the unit shift vectors for each edge in terms of
                    unit cell repetitions. These are integer values indicating how many unit cells to move in each
                    direction (x, y, z) to find the periodic image of the neighbor atom.
        cell: NumPy array of shape [3, 3] containing the unit cell vectors that define the simulation box.
    """
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell


# Andreas wrapper around get_neighborhood
def update_neighborhood_graph_batched(
    batch: dict, r_max: float, overwrite_cell: bool = True
) -> dict:
    """
    Get the neighborhood of each atom in the batch.
    """
    if isinstance(r_max, torch.Tensor) or isinstance(r_max, np.ndarray):
        r_max = r_max.item()
    # tempoary storage for edge_index, shifts, unit_shifts, cell
    edge_index_list = []
    shifts_list = []
    unit_shifts_list = []
    cell_list = []
    n_edges_list = []
    # loop over batches in superbatch
    # better to do via
    # positions_list = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0) # [B, N_atoms, 3]
    cnt = 0
    bs = batch["batch"].max() + 1
    for i in range(bs):
        n_atoms = torch.sum(batch["batch"] == i).item()
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=batch["positions"][cnt : cnt + n_atoms]
            .detach()
            .cpu()
            .numpy(),  # [N_atoms, 3]
            cutoff=r_max,
            cell=batch["cell"][i * 3 : (i + 1) * 3].detach().cpu().numpy(),  # [3, 3]
        )
        # shift edge_index by the n_atoms in previous batches
        edge_index += cnt
        edge_index_list.append(tensor_like(edge_index, batch["edge_index"]))
        shifts_list.append(tensor_like(shifts, batch["shifts"]))
        unit_shifts_list.append(tensor_like(unit_shifts, batch["unit_shifts"]))
        cell_list.append(tensor_like(cell, batch["cell"]))
        # n_edges_list.append(torch.tensor(edge_index.shape[1], device=batch["n_edges"].device, dtype=batch["n_edges"].dtype))
        cnt += n_atoms
    batch["edge_index"] = torch.cat(edge_index_list, dim=1)  # [2, B*N_edges]
    batch["shifts"] = torch.cat(shifts_list, dim=0)  # [B*N_edges, 3]
    batch["unit_shifts"] = torch.cat(unit_shifts_list, dim=0)  # [B*N_edges, 3]
    if overwrite_cell:
        batch["cell"] = torch.cat(cell_list, dim=0)  # [B*3, 3]
    return batch


###################################################################################################################
# torch version by Andreas
###################################################################################################################


# TODO: PBC / cell not tested! Works for non-periodic systems (molecules like alanine dipeptide)
def neighbour_list_torch(
    positions: torch.Tensor,  # [num_positions, 3]
    cell: torch.Tensor,  # [3, 3]
    cutoff: float,
    pbc: Tuple[bool, bool, bool] = (False, False, False),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of matscipy's neighbour_list function.

    Args:
        positions: Tensor of shape [num_positions, 3] containing atomic coordinates
        cell: Tensor of shape [3, 3] defining the unit cell vectors
        cutoff: Float value defining the maximum distance for considering atoms as neighbors
        pbc: Tuple of 3 booleans indicating periodic boundary conditions in x, y, z directions

    Returns:
        sender: Tensor of shape [n_edges] containing indices of sender atoms
        receiver: Tensor of shape [n_edges] containing indices of receiver atoms
        unit_shifts: Tensor of shape [n_edges, 3] containing the unit shift vectors
    """
    num_atoms = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Create all possible atom pairs
    senders = torch.arange(num_atoms, device=device).repeat_interleave(num_atoms)
    receivers = torch.arange(num_atoms, device=device).repeat(num_atoms)

    # Calculate distance vectors without PBC
    pos_i = positions[senders]
    pos_j = positions[receivers]
    distance_vectors = pos_j - pos_i  # [n_pairs, 3]

    # Initialize unit shifts
    unit_shifts = torch.zeros((len(senders), 3), device=device, dtype=dtype)

    # Apply PBC if needed
    if any(pbc):
        # Get inverse of cell for computing minimum image convention
        inv_cell = torch.linalg.inv(cell)

        # Convert distance vectors to fractional coordinates
        frac_dist = torch.matmul(distance_vectors, inv_cell)

        # Apply minimum image convention for periodic dimensions
        for dim in range(3):
            if pbc[dim]:
                # Round to nearest integer and subtract to get minimum image
                shifts = torch.round(frac_dist[:, dim])
                frac_dist[:, dim] -= shifts
                unit_shifts[:, dim] = -shifts  # Store the unit cell shifts

        # Convert back to Cartesian coordinates
        distance_vectors = torch.matmul(frac_dist, cell)

    # Calculate distances
    distances = torch.norm(distance_vectors, dim=1)

    # Filter pairs within cutoff
    mask = distances < cutoff

    # Remove self-interactions (i==j and no shifts)
    self_interaction = (senders == receivers) & torch.all(unit_shifts == 0, dim=1)
    mask = mask & ~self_interaction

    # TODO: not vmap-compatible
    return senders[mask], receivers[mask], unit_shifts[mask]


def get_neighborhood_torch(
    positions: torch.Tensor,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[torch.Tensor] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch version of get_neighborhood function.

    Args:
        positions: Tensor of shape [num_positions, 3] containing atomic coordinates
        cutoff: Float value defining the maximum distance for considering atoms as neighbors
        pbc: Optional tuple of 3 booleans indicating periodic boundary conditions in x, y, z directions
        cell: Optional 3×3 tensor defining the unit cell vectors
        true_self_interaction: Boolean flag to control whether self-interactions are included

    Returns:
        edge_index: Tensor of shape [2, n_edges] containing indices of connected atoms (sender, receiver)
        shifts: Tensor of shape [n_edges, 3] containing the shift vectors for each edge in Cartesian coordinates
        unit_shifts: Tensor of shape [n_edges, 3] containing the unit shift vectors for each edge
        cell: Tensor of shape [3, 3] containing the unit cell vectors
    """
    device = positions.device
    dtype = positions.dtype

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or torch.all(cell == 0):
        cell = torch.eye(3, device=device, dtype=dtype)

    assert len(pbc) == 3
    assert cell.shape == (3, 3)

    pbc_x, pbc_y, pbc_z = pbc
    identity = torch.eye(3, device=device, dtype=dtype)
    max_positions = torch.max(torch.abs(positions)) + 1

    # Make a copy of cell to avoid modifying the input
    cell = cell.clone()

    # Extend cell in non-periodic directions
    # Use out-of-place operation instead of inplace to be vmap-compatible
    if not pbc_x:
        # cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
        cell = torch.cat(
            [(max_positions * 5 * cutoff * identity[0, :]).unsqueeze(0), cell[1:, :]],
            dim=0,
        )
    if not pbc_y:
        cell = torch.cat(
            [
                cell[0:1, :],
                (max_positions * 5 * cutoff * identity[1, :]).unsqueeze(0),
                cell[2:, :],
            ],
            dim=0,
        )
    if not pbc_z:
        cell = torch.cat(
            [cell[0:2, :], (max_positions * 5 * cutoff * identity[2, :]).unsqueeze(0)],
            dim=0,
        )

    sender, receiver, unit_shifts = neighbour_list_torch(
        positions=positions,
        cell=cell,
        cutoff=cutoff,
        pbc=pbc,
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= torch.all(unit_shifts == 0, dim=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = torch.stack((sender, receiver))  # [2, n_edges]

    # Calculate shifts from unit_shifts and cell
    shifts = torch.matmul(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell


def update_neighborhood_graph_torch(
    batch: dict, r_max: float, overwrite_cell: bool = True
) -> dict:
    """
    Get the neighborhood of each atom in the batch.
    """
    edge_index, shifts, unit_shifts, cell = get_neighborhood_torch(
        positions=batch["positions"],
        cell=batch["cell"],
        cutoff=r_max,
        pbc=batch["pbc"],
    )
    batch["edge_index"] = edge_index
    batch["shifts"] = shifts
    batch["unit_shifts"] = unit_shifts
    if overwrite_cell:
        batch["cell"] = cell
    return batch


def update_neighborhood_graph_torch_batched(
    batch: dict, r_max: float, overwrite_cell: bool = True
) -> dict:
    """
    Get the neighborhood of each atom in the batch.
    """
    # tempoary storage for edge_index, shifts, unit_shifts, cell
    edge_index_list = []
    shifts_list = []
    unit_shifts_list = []
    cell_list = []
    n_edges_list = []
    # loop over batches in superbatch
    # better to do via
    # positions_list = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0) # [B, N_atoms, 3]
    cnt = 0
    bs = batch["batch"].max() + 1
    for i in range(bs):
        n_atoms = torch.sum(batch["batch"] == i).item()
        edge_index, shifts, unit_shifts, cell = get_neighborhood_torch(
            positions=batch["positions"][cnt : cnt + n_atoms],
            cell=batch["cell"][i * 3 : (i + 1) * 3],
            cutoff=r_max,
            pbc=batch["pbc"],
        )
        # shift edge_index by the n_atoms in previous batches
        edge_index += cnt
        edge_index_list.append(edge_index)
        shifts_list.append(shifts)
        unit_shifts_list.append(unit_shifts)
        cell_list.append(cell)
        # n_edges_list.append(torch.tensor(edge_index.shape[1], device=batch["n_edges"].device, dtype=batch["n_edges"].dtype))
        cnt += n_atoms
    batch["edge_index"] = torch.cat(edge_index_list, dim=1)  # [2, B*N_edges]
    batch["shifts"] = torch.cat(shifts_list, dim=0)  # [B*N_edges, 3]
    batch["unit_shifts"] = torch.cat(unit_shifts_list, dim=0)  # [B*N_edges, 3]
    if overwrite_cell:
        batch["cell"] = torch.cat(cell_list, dim=0)  # [B*3, 3]
    return batch


if __name__ == "__main__":
    positions = np.random.rand(10, 3)
    edge_index, shifts, unit_shifts, cell = get_neighborhood(positions, 1.0)
    print(edge_index)
    print(shifts)
    print(unit_shifts)
    print(cell)
