from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory

import numpy as np
import torch
import sys
import glob
import toml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import copy

import itertools
import time
import os
import json

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms

from scripts.md_run import setup_seed
from scripts.train import get_arguments, update_namespace, split_data
from sampler.samplermodel import SamplerModel
from sampler.build_mol_graph import add_connectivity_batch


def main():
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
        update_namespace(args, params)
    
    print(f"-"*80)
    
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
    
    # meta sampler
    sampler = SamplerModel(hidden_state_size=args.node_size)
    sampler.to(device)
    print(f"sampler parameters: {sum(p.numel() for p in sampler.parameters())}")
    
    # optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(sampler.parameters(), lr=args.initial_lr)
    criterion = torch.nn.MSELoss()
    if args.plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    else:
        scheduler_fn = lambda step: 0.96 ** (step / 100000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
    # early_stop = EarlyStopping(patience=args.stop_patience)    

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
    if args.overfit_to_single_sample:
        firstbatch = next(iter(train_loader))
        
        pbar = tqdm(range(args.max_steps))
        for _ in pbar:
            # If non_blocking=True, no synchronization is triggered, 
            # from the host perspective, multiple tensors can be sent to the device simultaneously, 
            # as the thread does not need to wait for one transfer to be completed to initiate the other.
            coord = firstbatch['coord'].to(device=device, non_blocking=True).requires_grad_(True)
            elems = firstbatch['elems'].to(device=device, non_blocking=True)
            cell = firstbatch['cell'].to(device=device, non_blocking=True).requires_grad_(True)
            # num_atoms = firstbatch['num_atoms'].to(device=device, non_blocking=True).float().requires_grad_(True)
            num_atoms = firstbatch['num_atoms'].to(device=device, non_blocking=True)
            # pairs = firstbatch['pairs'].to(device=device, non_blocking=True)
            # n_diff = firstbatch['n_diff'].to(device=device, non_blocking=True).requires_grad_(True)
            # num_pairs = firstbatch['num_pairs'].to(device=device, non_blocking=True)
            
            optimizer.zero_grad()
            
            coord_pred = sampler(
                num_atoms=num_atoms, elems=elems, cell=cell, coord=coord,
            )
            
            pairs, n_diff, num_pairs = add_connectivity_batch(
                num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
            )
            
            # print(f"coord_pred: \n{coord_pred}")
            # print(f"coord: \n{coord}")
            # print(f"cell: \n{cell}")
            
            # does not depend on coord, but on n_diff
            outputs = mlff(
                num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
                pairs=pairs, n_diff=n_diff, num_pairs=num_pairs,
                compute_forces=False,
                # compute_forces=bool(args.forces_weight)
            )
            
            loss = torch.sum(outputs["energy"])
            
            loss.backward()
            tqdm.write(f"loss: {loss.item():.1f}")
            
            # View gradients in each part of the sampler
            # for i, (name, param) in enumerate(sampler.named_parameters()):
            #     if param.grad is None:
            #         print(name, "grad is None, requires_grad : ", param.requires_grad)
            #     else:
            #         print (name, "grad", param.grad.data)
            
            # clip gradient
            grad = torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=10.0)
            optimizer.step()
            
            elapsed = time.time() - starttime
            pbar.set_description(f"loss: {loss.item():.1f}, grad: {torch.linalg.norm(grad):.1f}, time: {elapsed:.0f}s")
            
    else:

        step = 0
        pbar = tqdm(range(args.max_steps))
        for epoch in itertools.count():
            for batch_host in train_loader:
                coord = batch_host['coord'].to(device=device, non_blocking=True).requires_grad_(True)
                elems = batch_host['elems'].to(device=device, non_blocking=True)
                cell = batch_host['cell'].to(device=device, non_blocking=True).requires_grad_(True)
                # num_atoms = batch_host['num_atoms'].to(device=device, non_blocking=True).float().requires_grad_(True)
                num_atoms = batch_host['num_atoms'].to(device=device, non_blocking=True)
                # pairs = batch_host['pairs'].to(device=device, non_blocking=True)
                # n_diff = batch_host['n_diff'].to(device=device, non_blocking=True).requires_grad_(True)
                # num_pairs = batch_host['num_pairs'].to(device=device, non_blocking=True)
                
                optimizer.zero_grad()
                
                coord_pred = sampler(
                    num_atoms=num_atoms, elems=elems, cell=cell, coord=coord,
                )
                
                pairs, n_diff, num_pairs = add_connectivity_batch(
                    num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
                )
                
                # print(f"coord_pred: \n{coord_pred}")
                # print(f"coord: \n{coord}")
                # print(f"cell: \n{cell}")
                
                # does not depend on coord, but on n_diff
                outputs = mlff(
                    num_atoms=num_atoms, elems=elems, cell=cell, coord=coord_pred,
                    pairs=pairs, n_diff=n_diff, num_pairs=num_pairs,
                    compute_forces=False,
                    # compute_forces=bool(args.forces_weight)
                )
                
                loss = torch.sum(outputs["energy"])
                
                loss.backward()
                tqdm.write(f"loss: {loss.item():.1f}")
                
                # View gradients in each part of the sampler
                # for i, (name, param) in enumerate(sampler.named_parameters()):
                #     if param.grad is None:
                #         print(name, "grad is None, requires_grad : ", param.requires_grad)
                #     else:
                #         print (name, "grad", param.grad.data)
                
                # clip gradient
                grad = torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=2.0)
                optimizer.step()
                
                elapsed = time.time() - starttime
                pbar.set_description(f"loss={loss.item():.1f}, grad={torch.linalg.norm(grad):.1f}, lr={optimizer.param_groups[0]['lr']:.1e}, time={elapsed:.0f}s")
                pbar.update(1)
                step += 1
                
                if not args.plateau_scheduler:
                    scheduler.step()

                if step >= args.max_steps:
                    logging.info("Max steps reached, exiting")
                    # torch.save(
                    #     {
                    #         # "model": mlff.state_dict(),
                    #         "meta_sampler": sampler.state_dict(),
                    #         "optimizer": optimizer.state_dict(),
                    #         "scheduler": scheduler.state_dict(),
                    #         "step": step,
                    #         "best_val_loss": best_val_loss,
                    #         "node_size": args.node_size,
                    #         "num_layer": args.num_interactions,
                    #         "cutoff": args.cutoff,
                    #     },
                    #     os.path.join(args.output_dir, "exit_model.pth"),
                    # )
                    sys.exit(0)
    

if __name__ == "__main__":
    main()