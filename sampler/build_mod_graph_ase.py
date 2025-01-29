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

from ase.atoms import Atoms
import asap3

"""Reconstruction of the AseDataReader in the dataloader/MD from PaiNN/data.py, 
but takes in torch.tensors instead of ASE atoms objects.
"""

def get_neighborlist(atoms, cutoff):        
    nl = asap3.FullNeighborList(cutoff, atoms)
    pair_i_idx = []
    pair_j_idx = []
    n_diff = []
    for i in range(len(atoms)):
        indices, diff, _ = nl.get_neighbors(i)
        pair_i_idx += [i] * len(indices) # local index of pair i
        pair_j_idx.append(indices) # local index of pair j
        n_diff.append(diff)

    pair_j_idx = np.concatenate(pair_j_idx)
    pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
    n_diff = np.concatenate(n_diff)
    return pairs, n_diff

# CPU version without gradients in numpy and C
def add_connectivity_cpu(coord, elems, cell, pbc=True, cutoff=5):
    """Add connectivity (pairs, n_diff, num_pairs) to a single sample.
    
    Args:
        atoms_data: batch of molecules
        cutoff: cutoff distance for neighbor list
    Returns:
        pairs: pairs of atoms [n_pairs, 2]
        n_diff: difference in positions between pairs [n_pairs, 3]
        num_pairs: number of pairs [1]
    """
    if cell.any():
        pairs, n_diff = get_neighborlist(
            Atoms(
                positions=coord.detach().cpu().numpy(),
                numbers=elems.cpu().numpy(),
                cell=cell.cpu().numpy(),
                pbc=pbc, # np.array([True, True, True]),
            ), 
            cutoff,
        )
    else:
        pairs, n_diff = get_neighborlist_simple(coord.detach().cpu().numpy(), cutoff)
        
    pairs = torch.from_numpy(pairs).to(device=device)
    n_diff = torch.from_numpy(n_diff).to(device=device)
    num_pairs = torch.tensor([pairs.shape[0]], device=device)
    
    return pairs, n_diff, num_pairs


def get_neighborlist_simple(coord, cutoff):
    """CPU: no gradients"""
    pos = coord
    # pos = atoms.get_positions()
    dist_mat = distance_matrix(pos, pos)
    mask = dist_mat < cutoff
    np.fill_diagonal(mask, False)        
    pairs = np.argwhere(mask)
    n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    return pairs, n_diff