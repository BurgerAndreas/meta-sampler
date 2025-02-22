# from
# https://github.com/ACEsuit/mace/blob/3e578b02e649a5b2ac8109fa857698fdc42cf842/mace/data/neighborhood.py

from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
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
        edge_index_list.append(
            torch.tensor(
                edge_index,
                device=batch["edge_index"].device,
                dtype=batch["edge_index"].dtype,
            )
        )
        shifts_list.append(
            torch.tensor(
                shifts, device=batch["shifts"].device, dtype=batch["shifts"].dtype
            )
        )
        unit_shifts_list.append(
            torch.tensor(
                unit_shifts,
                device=batch["unit_shifts"].device,
                dtype=batch["unit_shifts"].dtype,
            )
        )
        cell_list.append(
            torch.tensor(cell, device=batch["cell"].device, dtype=batch["cell"].dtype)
        )
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
