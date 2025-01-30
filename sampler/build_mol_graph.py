import numpy as np
import math
import json, os, sys, toml
from pathlib import Path
import argparse
import logging
import itertools
import torch
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

from ase.io import read, write, Trajectory
from PaiNN.data import AseDataset, collate_atomsdata, cat_tensors, AseDataReader
from PaiNN.model import PainnModel
from ase.atoms import Atoms
import torch_cluster


from scripts.train import get_arguments, update_namespace, setup_seed, split_data
from sampler.build_mod_graph_ase import get_neighborlist

"""Compute molecular graph (add edges) after predicting atom coordinates.

Batch in the dataset contains:
- num_atoms: number of atoms in the batch[B] # use
- elems: atom type Z [n_atoms_batch]. n_atoms_batch = sum(num_atoms) # use
- cell: describes periodic structure of the crystal and the adsorbate, liquid [B*3, 3, 3] # use
- coord: atom positions [n_atoms_batch, 3] # predict
- pairs: pairs of atoms [n_pairs_batch, 2]. n_pairs_batch = sum(num_pairs) # recompute after prediction
- n_diff: difference in positions between pairs [n_pairs_batch, 3] # recompute after prediction
- num_pairs: number of pairs in the batch [B] # recompute after prediction
- energy: potential energy [B] # ignore
- forces: forces [n_atoms_batch, 3] # ignore
"""

def get_neighborlist_torch(coord, cell, pbc, cutoff, device=None):
    """
    Constructs a neighbor list for a set of atomic positions, with periodic boundary conditions.
    Gives the same result as ASE/asap3, but ensures proper gradients flow through all tensors.

    Args:
        positions (torch.Tensor): Tensor of shape (N, 3) representing atomic positions.
        cell (torch.Tensor): Tensor of shape (3, 3) representing the simulation cell vectors.
        pbc (torch.Tensor): Tensor of shape (3,) indicating periodic boundary conditions in x, y, z (bool).
        cutoff (float): Cutoff distance for defining neighbors.

    Returns:
        pairs (torch.Tensor): Tensor of shape (M, 2) representing pairs of neighbor indices [i, j].
        n_diff (torch.Tensor): Tensor of shape (M, 3) representing displacement vectors between pairs.
    """
    positions = coord
    device = positions.device
    num_atoms = positions.shape[0]
    cutoff_squared = cutoff ** 2
    
    if isinstance(pbc, bool):
        pbc = torch.tensor([pbc] * 3, device=device)
    
    # Compute fractional coordinates relative to the simulation cell
    inv_cell = torch.linalg.inv(cell)  # Inverse of the cell matrix
    fractional_positions = torch.matmul(positions, inv_cell.T)
    
    # Apply periodic boundary conditions: map fractional coordinates into [0, 1)
    if torch.any(pbc):
        fractional_positions = fractional_positions - torch.floor(fractional_positions)
    
    # Map back to Cartesian coordinates
    positions = torch.matmul(fractional_positions, cell.T)

    # Compute pairwise squared distances efficiently
    # Get unique pairs, avoid self-pairs and duplicates
    row_i, row_j = torch.triu_indices(num_atoms, num_atoms, 1, device=device)  
    diff = positions[row_i] - positions[row_j]  # Compute displacements directly

    # Apply periodic boundary conditions to displacement vectors
    if torch.any(pbc):
        fractional_diff = torch.matmul(diff, inv_cell.T)  # Convert to fractional coordinates
        fractional_diff = fractional_diff - torch.round(fractional_diff)  # Wrap into [-0.5, 0.5)
        diff = torch.matmul(fractional_diff, cell.T)  # Convert back to Cartesian coordinates
    
    # Compute squared distances
    squared_distances = torch.sum(diff**2, dim=-1)  # Shape: (M,)

    # Mask to find neighbors within the cutoff distance
    neighbor_mask = squared_distances <= cutoff_squared
    # neighbor_mask = torch.where(squared_distances <= cutoff_squared)[0]
    
    # Extract valid pairs and corresponding displacements
    pair_i_idx = row_i[neighbor_mask]
    pair_j_idx = row_j[neighbor_mask]
    n_diff = diff[neighbor_mask]  # Only keep displacements of valid pairs
    
    # Ensure symmetry by adding reverse pairs
    # every neighbor pair appears in both directions (i,ji,j and j,ij,i)
    pair_i_idx_symmetric = torch.cat((pair_i_idx, pair_j_idx), dim=0)
    pair_j_idx_symmetric = torch.cat((pair_j_idx, pair_i_idx), dim=0)
    n_diff_symmetric = torch.cat((n_diff, -n_diff), dim=0)

    # Stack pair indices into a single tensor
    pairs = torch.stack((pair_i_idx_symmetric, pair_j_idx_symmetric), dim=1)  # Shape: (M, 2)
    
    return pairs, n_diff_symmetric


def get_neighborlist_simple_torch(coord, cutoff, device):
    """Torch: GPU and gradients"""
    pos = coord
    # Calculate pairwise distances using torch operations
    dist_mat = torch.cdist(pos, pos, device=device)
    # Create mask for distances less than cutoff
    mask = dist_mat < cutoff
    # Zero out diagonal
    mask.fill_diagonal_(False)
    # Get pairs of indices where mask is True
    pairs = torch.nonzero(mask, device=device)
    # Calculate position differences for the pairs
    n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    return pairs, n_diff



# GPU version with gradients
def add_connectivity_torch(coord, elems, cell, pbc=True, cutoff=5):
    """Add connectivity (pairs, n_diff, num_pairs) to a single sample.
    
    Args:
        atoms_data: batch of molecules
        cutoff: cutoff distance for neighbor list
    Returns:
        pairs: pairs of atoms [n_pairs, 2]
        n_diff: difference in positions between pairs [n_pairs, 3]
        num_pairs: number of pairs [1]
    """
    device = coord.device
    if cell.any():
        pairs, n_diff = get_neighborlist_torch(
            coord=coord, cell=cell, pbc=pbc, cutoff=cutoff, device=device
        )
    else:
        pairs, n_diff = get_neighborlist_simple_torch(coord, cutoff, device=device)
    num_pairs = torch.tensor([pairs.shape[0]], device=device)
    return pairs, n_diff, num_pairs

def add_connectivity_batch(num_atoms, elems, cell, coord, cutoff=5):
    """Add connectivity (pairs, n_diff, num_pairs) to a batch of molecules.
    
    Args:
        batch: batch of molecules
        cutoff: cutoff distance for neighbor list
    Returns:
        pairs: pairs of atoms [n_pairs, 2]
        n_diff: difference in positions between pairs [n_pairs, 3]
        num_pairs: number of pairs [1]
    """
    device = coord.device
    
    all_pairs = []
    all_n_diff = []
    all_num_pairs = []
    # split batch into individual samples
    prev_atoms = 0
    for b in range(len(num_atoms)):
        _num_atoms = num_atoms[b]
        _elems = elems[prev_atoms:prev_atoms+_num_atoms]
        _coord = coord[prev_atoms:prev_atoms+_num_atoms]
        _cell = cell[b:b+3]
        
        prev_atoms += _num_atoms
        
        pairs, n_diff, num_pairs = add_connectivity_torch(
            coord=_coord, elems=_elems, cell=_cell, cutoff=cutoff
        )
        
        all_pairs.append(pairs)
        all_n_diff.append(n_diff)
        all_num_pairs.append(num_pairs)
    
    # merge all samples
    all_pairs = torch.cat(all_pairs)
    all_n_diff = torch.cat(all_n_diff)
    all_num_pairs = torch.cat(all_num_pairs)
    
    return all_pairs, all_n_diff, all_num_pairs



if __name__ == "__main__":
    
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)
    
    args.batch_size = 4
    print(f"-"*80)

    # Setup random seed
    setup_seed(args.random_seed)

    # Create device
    device = torch.device(args.device)

    # Setup dataset and loader
    print(f"loading data {args.dataset}")
    dataset = AseDataset(
        args.dataset,
        cutoff = args.cutoff,
    )

    datasplits = split_data(dataset, args)

    train_loader = torch.utils.data.DataLoader(
        datasplits["train"],
        args.batch_size,
        sampler=torch.utils.data.RandomSampler(datasplits["train"]),
        collate_fn=collate_atomsdata,
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"], 
        args.batch_size, 
        collate_fn=collate_atomsdata,
    )
    
    print(f"Dataset size: {len(dataset)}, training set size: {len(datasplits['train'])}, validation set size: {len(datasplits['validation'])}")

    # symbols='H60Au36O32, symbols='H114Au64O59'
    # print(f"Example sample: {dataset[0]}")
    # print(f"Example sample: {dataset[-1]}")

    ############################################################################
    
    ############################################################################
    # make sure gradients can flow through get_neighborlist_torch
    print(f"-"*80)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    coord = torch.randn(10, 3, requires_grad=True, device=device)
    cell = torch.randn(3, 3, requires_grad=False, device=device)
    pairs, n_diff = get_neighborlist_torch(coord, cell, pbc=True, cutoff=5, device=device)
    loss = torch.sum(n_diff) + torch.sum(pairs)
    loss.backward()
    #  Should not be None or zeros
    print(f"grad of coord: {coord.grad}") 
    print(f"grad of cell: {cell.grad}")
        
    ############################################################################
    # Ensure that the coords are inside the cell
    
    # take first batch
    batch = next(iter(train_loader))
    
    pairs, n_diff, num_pairs = add_connectivity_batch(
        num_atoms=batch['num_atoms'],
        elems=batch['elems'],
        cell=batch['cell'],
        coord=batch['coord'],
        cutoff=5,
    )
    
    # split batch into individual samples
    prev_atoms = 0
    for b, num_atoms in enumerate(batch['num_atoms']):
        _num_atoms = num_atoms
        _elems = batch['elems'][prev_atoms:prev_atoms+_num_atoms]
        _coord = batch['coord'][prev_atoms:prev_atoms+_num_atoms]
        _cell = batch['cell'][b:b+3]
        prev_atoms += num_atoms
        
        print(f"cell: \n{_cell}")
        
        # check if coords are inside the cell
        away_from_center = torch.linalg.norm(_coord, dim=1)
        print(f"away from center: {away_from_center.shape}")
        print(f"away from center: {away_from_center.max()}")
        
        # 3d plot of cell
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(_coord[:, 0], _coord[:, 1], _coord[:, 2])
        # # plot the cell
        # ax.plot([0, _cell[0, 0]], [0, _cell[1, 0]], [0, _cell[2, 0]], color='black')
        # ax.plot([0, _cell[0, 1]], [0, _cell[1, 1]], [0, _cell[2, 1]], color='black')
        # ax.plot([0, _cell[0, 2]], [0, _cell[1, 2]], [0, _cell[2, 2]], color='black')
        # fname = f"cell_{b}.png"
        # plt.savefig(fname)
        # print(f"saved {fname}")
        
        # check if coords are inside the cell
        cellmax = _cell.max(dim=1).values
        inside_cell = torch.all(_coord < cellmax, dim=1)
        print(f"cellmax: {cellmax}")
        print(f"coord: {_coord.shape}")
        print(f"inside cell: {inside_cell.shape}")
        print(f"inside cell: {inside_cell.max()}")
        break
        
    ############################################################################
    
    # Make sure add_connectivity is the same CPU (ASE) / torch (homemade)
    
    # split batch into individual samples
    prev_atoms = 0
    for b, num_atoms in enumerate(batch['num_atoms']):
        _num_atoms = num_atoms
        _elems = batch['elems'][prev_atoms:prev_atoms+_num_atoms]
        _coord = batch['coord'][prev_atoms:prev_atoms+_num_atoms]
        _cell = batch['cell'][b:b+3]
        prev_atoms += num_atoms
        
        pairs, n_diff  = get_neighborlist_torch(_coord, cell=_cell, pbc=True, cutoff=5, device=device)
        
        pairs2, n_diff2 = get_neighborlist(
            Atoms(
                positions=_coord.detach().cpu().numpy(),
                numbers=_elems.cpu().numpy(),
                cell=_cell.cpu().numpy(),
                pbc=True,
            ), 
            cutoff=5,
        )
    
        print("")
        # sort pairs and n_diff to match numpy version
        pairs = pairs.detach().cpu().numpy()
        n_diff = n_diff.detach().cpu().numpy()
        # Sort pairs and pairs2 by first column, then second column
        pairs = pairs[np.lexsort((pairs[:,1], pairs[:,0]))]
        pairs2 = pairs2[np.lexsort((pairs2[:,1], pairs2[:,0]))]
        print(f"pairs allclose: {np.allclose(pairs, pairs2)}")
        print(f"delta in pairs: {np.max(np.abs(pairs - pairs2))}")
        
        # Sort n_diff [num_edges, 3]
        print("")
        assert n_diff.shape == n_diff2.shape, f"n_diff: {n_diff.shape}, n_diff2: {n_diff2.shape}"
        n_diff = n_diff[np.lexsort((n_diff[:,2], n_diff[:,1], n_diff[:,0]))]
        n_diff2 = n_diff2[np.lexsort((n_diff2[:,2], n_diff2[:,1], n_diff2[:,0]))]
        print(f"n_diff={n_diff.shape}, n_diff2={n_diff2.shape}")
        # there are some rounding errors in the n_diff values
        # which leads to different sorting
        print(f"max diff n_diff: {np.max(np.abs(n_diff - n_diff2))}")
        # number of indices where difference is greater than 1e-3
        print(f"number of indices where diff > 1e-3: {(np.abs(n_diff - n_diff2) > 1e-3).sum()}")
        # indices where difference is greater than 1e-3
        diff_idx = np.where(np.abs(n_diff - n_diff2) > 1e-3) # tuple of arrays
        print(f"diff_idx: {len(diff_idx)}, {len(diff_idx[0])}")
        # print(f"rows where diff > 1e-3:\n {n_diff[diff_idx[0]]}")
        # print(f"rows where diff > 1e-3:\n {n_diff2[diff_idx[0]]}")
        # Compare n_diff values for common pairs
        sum_deltas = 0
        for i in range(diff_idx[0].shape[0]):
            idx = diff_idx[0][i]
            # Calculate distances between this n_diff vector and all n_diff2 vectors
            distances = np.linalg.norm(n_diff2 - n_diff[idx], axis=1)
            closest_idx = np.argmin(distances)
            # print(f"idx: {idx}, closest_idx: {closest_idx}, distance: {distances[closest_idx]:.6f}")
            sum_deltas += np.linalg.norm(n_diff[idx] - n_diff2[closest_idx])
        
        print(f"sum of deltas: {sum_deltas}")
        print(f"mean of deltas: {sum_deltas / diff_idx[0].shape[0]}")
        
        break
        
    