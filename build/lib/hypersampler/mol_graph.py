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
import asap3
import numpy as np
from ase.io import read, write, Trajectory
from scipy.spatial import distance_matrix
from PaiNN.data import AseDataset, collate_atomsdata, AseDataReader
from PaiNN.model import PainnModel
from ase.atoms import Atoms

from scripts.train import get_arguments, update_namespace, setup_seed, split_data

"""Compute molecular graph (add edges) after predicting atom coordinates.

Batch in the dataset contains:
- num_atoms: number of atoms in the batch[B] # use
- elems: atom type Z [n_atoms_batch]. n_atoms_batch = sum(num_atoms) # use
- cell: cell [B*3, 3, 3] # use
- coord: atom positions [n_atoms_batch, 3] # predict
- pairs: pairs of atoms [n_pairs_batch, 2]. n_pairs_batch = sum(num_pairs) # recompute after prediction
- n_diff: difference in positions between pairs [n_pairs_batch, 3] # recompute after prediction
- num_pairs: number of pairs in the batch [B] # recompute after prediction
- energy: potential energy [B] # ignore
- forces: forces [n_atoms_batch, 3] # ignore
"""
       
def add_connectivity(atoms_data, cutoff=5):
    # make a copy as ASE atom object
    atoms = Atoms(
        positions=atoms_data['coord'],
        numbers=atoms_data['elems'],
        cell=atoms_data['cell'],
    )
    
    if atoms_data['cell'].any():
        pairs, n_diff = get_neighborlist(atoms, cutoff)
        
    else:
        pairs, n_diff = get_neighborlist_simple(atoms, cutoff)
        
    atoms_data['pairs'] = torch.from_numpy(pairs)
    atoms_data['n_diff'] = torch.from_numpy(n_diff).float()
    atoms_data['num_pairs'] = torch.tensor([pairs.shape[0]])
    
    return atoms_data
        

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

def get_neighborlist_simple(atoms, cutoff):
    pos = atoms['coord']
    # pos = atoms.get_positions()
    dist_mat = distance_matrix(pos, pos)
    mask = dist_mat < cutoff
    np.fill_diagonal(mask, False)        
    pairs = np.argwhere(mask)
    n_diff = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    return pairs, n_diff

def add_connectivity_batch(batch, cutoff=5):
    """Add connectivity (pairs, n_diff, num_pairs) to a batch of molecules.
    
    Args:
        batch: batch of molecules
        cutoff: cutoff distance for neighbor list
    Returns:
        newbatch: batch of molecules with connectivity
    """
    newbatch = {
        "num_atoms": [],
        "elems": [],
        "coord": [],
        "cell": [],
        "pairs": [],
        "n_diff": [],
        "num_pairs": [],
    }
    # split batch into individual samples
    prev_atoms = 0
    for b, num_atoms in enumerate(batch['num_atoms']):
        _sample = {
            "num_atoms": num_atoms,
            "elems": batch['elems'][prev_atoms:prev_atoms+num_atoms],
            "coord": batch['coord'][prev_atoms:prev_atoms+num_atoms],
            "cell": batch['cell'][b:b+3],
        }
        prev_atoms += num_atoms
        
        _samplen = add_connectivity(_sample, cutoff=cutoff)
        
        for k in newbatch.keys():
            newbatch[k].append(_samplen[k])
    
    # merge all samples
    newbatch['num_atoms'] = torch.hstack(newbatch['num_atoms'])
    newbatch['elems'] = torch.hstack(newbatch['elems'])
    newbatch['cell'] = torch.cat(newbatch['cell'], axis=0)
    newbatch['coord'] = torch.cat(newbatch['coord'], axis=0)
    newbatch['pairs'] = torch.cat(newbatch['pairs'], axis=0)
    newbatch['n_diff'] = torch.cat(newbatch['n_diff'], axis=0)
    newbatch['num_pairs'] = torch.hstack(newbatch['num_pairs'])
    
    return newbatch

if __name__ == "__main__":
    
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)
    
    print(f"-"*80)

    # Setup random seed
    setup_seed(args.random_seed)

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )
    
    current_dir = os.getcwd()
    args.output_dir = os.path.join(current_dir, args.output_dir)
    print(f"Output directory: {args.output_dir}")

    # Save command line args
    with open(os.path.join(args.output_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(args.output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Create device
    device = torch.device(args.device)

    # Setup dataset and loader
    logging.info("loading data %s", args.dataset)
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
    
    logging.info('Dataset size: {}, training set size: {}, validation set size: {}'.format(
        len(dataset),
        len(datasplits["train"]),
        len(datasplits["validation"]),
    ))

    ############################################################################
    # take first sample
    batch = next(iter(train_loader))
    
    # newbatch = add_connectivity_batch(batch, cutoff=5.8)
    newbatch = add_connectivity_batch(batch, cutoff=5)
    
    for k in newbatch.keys():
        print(f"{k}: {newbatch[k].shape} vs {batch[k].shape}")
        
    # newbatch2 = add_connectivity(batch)
    # for k in newbatch2.keys():
    #     print(f"{k}: {newbatch2[k].shape} vs {batch[k].shape}")
   
    
    