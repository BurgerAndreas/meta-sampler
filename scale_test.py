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
from sampler.build_mol_graph import add_connectivity_batch


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

# How much can we perturb the coordinates before the MLFF outputs inf/nan?
# On what scale does the offset predictor output need to be?


def scale_test_main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)
    
    print(f"-"*80)
    # hard code
    # --load_model trained_models/96_node_3_layer.pth --batch_size 2
    args.load_model = "trained_models/96_node_3_layer.pth"
    args.batch_size = 4
    
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
    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    # torch.tensor([0], device=device)

    ######################################################################
    # Setup dataset and loader
    ######################################################################
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
        pin_memory=True, # True will be faster, unless you run out of RAM
    )
    val_loader = torch.utils.data.DataLoader(
        datasplits["validation"], 
        args.batch_size, 
        collate_fn=collate_atomsdata,
        pin_memory=True, # True will be faster, unless you run out of RAM
    )
    
    logging.info('Dataset size: {}, training set size: {}, validation set size: {}'.format(
        len(dataset),
        len(datasplits["train"]),
        len(datasplits["validation"]),
    ))

    if args.normalization:
        logging.info("Computing mean and variance")
        target_mean, target_stddev = get_normalization(
            datasplits["train"], 
            per_atom=args.atomwise_normalization,
        )
        logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    ######################################################################
    # Setup models (MLFF and meta sampler)
    ######################################################################
    
    mlff = PainnModel(
        num_interactions=args.num_interactions, 
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        normalization=args.normalization,
        target_mean=target_mean.tolist() if args.normalization else [0.0],
        target_stddev=target_stddev.tolist() if args.normalization else [1.0],
        atomwise_normalization=args.atomwise_normalization,
    )
    mlff.to(device)
    
    n_params = sum(p.numel() for p in mlff.parameters())
    print(f"MLFF parameters: {n_params}")
    
    if args.load_model:
        logging.info(f"Load model from {args.load_model}")
        state_dict = torch.load(args.load_model, weights_only=False)
        mlff.load_state_dict(state_dict["model"])
        # step = state_dict["step"]
        # best_val_loss = state_dict["best_val_loss"]
        # optimizer.load_state_dict(state_dict["optimizer"])
        # scheduler.load_state_dict(state_dict["scheduler"])
    
    ######################################################################
    # train meta sampler
    ######################################################################
    
    starttime = time.time()
    # fit to a single batch
    for recompute_graph in [True, False]:
        for noise_scale in torch.linspace(0.0, 1.0, 100):
            firstbatch = next(iter(train_loader))
            
            # If non_blocking=True, no synchronization is triggered, 
            # from the host perspective, multiple tensors can be sent to the device simultaneously, 
            # as the thread does not need to wait for one transfer to be completed to initiate the other.
            coord = firstbatch['coord'].to(device=device, non_blocking=True).requires_grad_(True)
            elems = firstbatch['elems'].to(device=device, non_blocking=True)
            cell = firstbatch['cell'].to(device=device, non_blocking=True).requires_grad_(True)
            num_atoms = firstbatch['num_atoms'].to(device=device, non_blocking=True)
            pairs = firstbatch['pairs'].to(device=device, non_blocking=True)
            n_diff = firstbatch['n_diff'].to(device=device, non_blocking=True).requires_grad_(True)
            num_pairs = firstbatch['num_pairs'].to(device=device, non_blocking=True)
            
            coord_pred = coord + noise_scale * torch.randn_like(coord)
            
            if recompute_graph:
                pairs, n_diff, num_pairs = add_connectivity_batch(
                    num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
                )
            
            # does not depend on coord, but on n_diff
            outputs = mlff(
                num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
                pairs=pairs, n_diff=n_diff, num_pairs=num_pairs,
                # compute_forces=bool(args.forces_weight)
                compute_forces=False
            )
            
            loss = torch.sum(outputs["energy"])
            print(f"noise={noise_scale:.2e}, loss={loss.item():.1f} {'(recomputed)' if recompute_graph else ''}")
                
if __name__ == "__main__":
    scale_test_main()
    