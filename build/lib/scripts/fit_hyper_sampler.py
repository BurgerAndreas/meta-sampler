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

import itertools
import time
import os
import json

from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms

from scripts.md_run import setup_seed, get_arguments
from hypersampler.model import HyperSampler
from hypersampler.mol_graph import add_connectivity_batch


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

    if args.normalization:
        logging.info("Computing mean and variance")
        target_mean, target_stddev = get_normalization(
            datasplits["train"], 
            per_atom=args.atomwise_normalization,
        )
        logging.debug("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

    
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
    
    # meta sampler
    n_params = sum(p.numel() for p in model.parameters())
    meta_sampler = MetaSampler(n_params)
    
    optimizer = torch.optim.Adam(metasampler.parameters(), lr=args.initial_lr)
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
    
    
    
    # train meta sampler
    
    optimizer = torch.optim.Adam(meta_sampler.parameters(), lr=1e-3)
    
    firstbatch = next(iter(train_loader))
    
    for _ in tqdm(range(1000)):
        batch = {
            k: v.to(device=device, non_blocking=True)
            for (k, v) in firstbatch.items()
        }
        optimizer.zero_grad()
        # pred = meta_sampler(batch, model.get_parameters())
        pred = meta_sampler(batch)
        # TODO: akward gradient flow
        pred = add_connectivity_batch(pred)
        outputs = mlff(
            pred, compute_forces=bool(args.forces_weight)
        )
        loss = torch.sum(outputs["energy"])
        loss.backward()
        optimizer.step()
        tqdm.write(f"loss: {loss.item()}")

    start = time.time()

    for epoch in itertools.count():
        for batch_host in tqdm(train_loader):
            # Transfer to 'device'
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            # Reset gradient
            optimizer.zero_grad()

            # Forward, backward and optimize
            pred = meta_sampler(batch)
            # TODO: akward gradient flow
            pred = add_connectivity_batch(pred)
            outputs = mlff(
                pred, compute_forces=bool(args.forces_weight)
            )
            loss = torch.sum(outputs["energy"])
            loss.backward()
            optimizer.step()
            
            step += 1

            if not args.plateau_scheduler:
                scheduler.step()

            if step >= args.max_steps:
                logging.info("Max steps reached, exiting")
                torch.save(
                    {
                        "model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "node_size": args.node_size,
                        "num_layer": args.num_interactions,
                        "cutoff": args.cutoff,
                    },
                    os.path.join(args.output_dir, "exit_model.pth"),
                )
                sys.exit(0)
    

if __name__ == "__main__":
    main()