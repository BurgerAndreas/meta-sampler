from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image

import copy
import mace
from mace.calculators import mace_off, mace_anicc
from mace.tools import torch_geometric, torch_tools, utils
import mace.data
import ase
import ase.io
import ase.build
from ase.calculators.calculator import Calculator, all_changes
import openmm
from openmm.unit import nanometer
from openmm.unit import Quantity

import torch_geometric as tg

from alanine_dipeptide.mace_neighbourhood import (
    update_neighborhood_graph_batched,
    get_neighborhood,
    get_neighborhood_torch,
)
from alanine_dipeptide.dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_vmap,
    set_dihedral_torch_batched,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
)
from alanine_dipeptide.alanine_dipeptide_mace import (
    load_alanine_dipeptide_ase,
    repeated_atoms_to_batch,
    update_alanine_dipeptide_xyz_from_dihedrals_batched,
    update_alanine_dipeptide_xyz_from_dihedrals_torch,
)


########################################################################################
# Helper functions
########################################################################################


def tensor_like(_new, _base):
    return torch.tensor(_new, device=_base.device, dtype=_base.dtype)


########################################################################################
# Tests
########################################################################################


def test_if_energy_depends_r_max(model, singlebatch_base):
    """Energy does not depend on r_max, even though it changes the number of edges.
    Maybe there is a decay in the distance in the message passing, that sets the long range interactions to zero?
    """
    print(f"-" * 80)
    print(f"edges before: {singlebatch_base['edge_index'].shape[1]}")

    # -------- Old r_max --------
    minibatch = singlebatch_base.to_dict()
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=minibatch["positions"].detach().cpu().numpy(),  # [N_atoms, 3]
        cutoff=model.r_max.item(),
        cell=minibatch["cell"].detach().cpu().numpy(),  # [3, 3]
    )
    minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
    minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
    minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
    minibatch["cell"] = tensor_like(cell, minibatch["cell"])

    result = model(minibatch, compute_stress=compute_stress, training=True)
    print(
        f"energy for r_max={model.r_max.item()}: {result['energy'].item():.4e} ({edge_index.shape[1]} edges)"
    )
    energy_old = result["energy"].item()

    # -------- New r_max --------
    r_max = model.r_max.item() * 1000  # big enough to include all edges
    minibatch = singlebatch_base.to_dict()
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=minibatch["positions"].detach().cpu().numpy(),  # [N_atoms, 3]
        cutoff=r_max,
        cell=minibatch["cell"].detach().cpu().numpy(),  # [3, 3]
    )
    minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
    minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
    minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
    minibatch["cell"] = tensor_like(cell, minibatch["cell"])

    result = model(minibatch, compute_stress=compute_stress, training=True)
    print(
        f"energy for r_max={r_max}: {result['energy'].item():.4e} ({edge_index.shape[1]} edges)"
    )
    energy_new = result["energy"].item()

    if energy_new != energy_old:
        print(
            f"Energy depends on r_max (relative difference: {np.abs(energy_new - energy_old) / energy_old:.4e}) ❌"
        )
    else:
        print(f"Energy does not depend on r_max ✅")
    return


def test_if_energy_depends_cell(model, singlebatch_base):
    """Energy does not depend on cell, probably because we don't have periodic boundary conditions."""
    print(f"-" * 80)
    print(f"edges before: {singlebatch_base['edge_index'].shape[1]}")
    print(f"cell before: {singlebatch_base['cell']}")

    # -------- Old cell --------
    minibatch = singlebatch_base.to_dict()
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=minibatch["positions"].detach().cpu().numpy(),  # [N_atoms, 3]
        cutoff=model.r_max.item(),
        cell=minibatch["cell"].detach().cpu().numpy(),  # [3, 3]
    )
    minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
    minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
    minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
    minibatch["cell"] = tensor_like(cell, minibatch["cell"])

    result = model(minibatch, compute_stress=compute_stress, training=True)
    print(
        f"energy for cell={cell}: {result['energy'].item():.4e} ({edge_index.shape[1]} edges)"
    )
    energy_old = result["energy"].item()

    # -------- New cell --------
    r_max = model.r_max.item()
    minibatch = singlebatch_base.to_dict()
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=minibatch["positions"].detach().cpu().numpy(),  # [N_atoms, 3]
        cutoff=r_max,
        # cell=minibatch["cell"].detach().cpu().numpy() * 1000,  # [3, 3]
        cell=None,
    )
    minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
    minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
    minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
    cell *= 1000
    minibatch["cell"] = tensor_like(cell, minibatch["cell"])

    result = model(minibatch, compute_stress=compute_stress, training=True)
    print(
        f"energy for cell={cell}: {result['energy'].item():.4e} ({edge_index.shape[1]} edges)"
    )
    energy_new = result["energy"].item()

    if energy_new != energy_old:
        print(
            f"Energy depends on cell (relative difference: {np.abs(energy_new - energy_old) / energy_old:.4e}) ❌"
        )
    else:
        print(f"Energy does not depend on cell ✅")
    return


def test_torch_vs_numpy_neighborhood(model, singlebatch_base):
    """Test if get_neighborhood_torch gives the same result as get_neighborhood."""
    print(f"-" * 80)
    print("Testing PyTorch vs NumPy neighborhood implementation")

    # Get the batch data
    minibatch = singlebatch_base.to_dict()
    positions = minibatch["positions"]
    cell = minibatch["cell"]
    cutoff = model.r_max.item()

    # NumPy implementation
    edge_index_np, shifts_np, unit_shifts_np, cell_np = get_neighborhood(
        positions=positions.detach().cpu().numpy(),
        cutoff=cutoff,
        cell=cell.detach().cpu().numpy(),
    )

    # PyTorch implementation
    edge_index_torch, shifts_torch, unit_shifts_torch, cell_torch = (
        get_neighborhood_torch(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
        )
    )

    # Convert NumPy results to PyTorch tensors for comparison
    edge_index_np_tensor = tensor_like(edge_index_np, edge_index_torch)
    shifts_np_tensor = tensor_like(shifts_np, shifts_torch)
    unit_shifts_np_tensor = tensor_like(unit_shifts_np, unit_shifts_torch)
    cell_np_tensor = tensor_like(cell_np, cell_torch)

    # Compare results
    # edge_index_np_tensor = torch.sort(edge_index_np_tensor, dim=0)[0]
    # edge_index_torch = torch.sort(edge_index_torch, dim=0)[0]
    # edge_index_match = torch.all(edge_index_np_tensor == edge_index_torch)
    edge_index_match = edge_index_np.shape[1] == edge_index_torch.shape[1]
    shifts_match = torch.allclose(shifts_np_tensor, shifts_torch, atol=1e-6)
    unit_shifts_match = torch.all(unit_shifts_np_tensor == unit_shifts_torch)
    cell_match = torch.allclose(cell_np_tensor, cell_torch, atol=1e-6)

    # Print results
    print(
        f"Edge indices match: {edge_index_match} {'✅' if edge_index_match else '❌'} (NumPy: {edge_index_np.shape}, PyTorch: {edge_index_torch.shape})"
    )
    print(
        f"Shifts match: {shifts_match} {'✅' if shifts_match else '❌'} (NumPy: {shifts_np.shape}, PyTorch: {shifts_torch.shape})"
    )
    print(
        f"Unit shifts match: {unit_shifts_match} {'✅' if unit_shifts_match else '❌'} (NumPy: {unit_shifts_np.shape}, PyTorch: {unit_shifts_torch.shape})"
    )
    print(f"Cell match: {cell_match} {'✅' if cell_match else '❌'}")

    # Check if energy calculations match
    minibatch_np = singlebatch_base.to_dict()
    minibatch_np["edge_index"] = tensor_like(edge_index_np, minibatch["edge_index"])
    minibatch_np["shifts"] = tensor_like(shifts_np, minibatch["shifts"])
    minibatch_np["unit_shifts"] = tensor_like(unit_shifts_np, minibatch["unit_shifts"])
    minibatch_np["cell"] = tensor_like(cell_np, minibatch["cell"])

    minibatch_torch = singlebatch_base.to_dict()
    minibatch_torch["edge_index"] = edge_index_torch
    minibatch_torch["shifts"] = shifts_torch
    minibatch_torch["unit_shifts"] = unit_shifts_torch
    minibatch_torch["cell"] = cell_torch

    result_np = model(minibatch_np, compute_stress=compute_stress, training=True)
    result_torch = model(minibatch_torch, compute_stress=compute_stress, training=True)

    energy_np = result_np["energy"].item()
    energy_torch = result_torch["energy"].item()

    energy_match = np.isclose(energy_np, energy_torch, rtol=1e-5)
    print(f"Energy from NumPy neighborhood: {energy_np:.6e}")
    print(f"Energy from PyTorch neighborhood: {energy_torch:.6e}")
    print(f"Energy values match: {energy_match}")

    if all(
        [edge_index_match, shifts_match, unit_shifts_match, cell_match, energy_match]
    ):
        print(
            "PyTorch and NumPy neighborhood implementations give identical results ✅"
        )
    else:
        print(
            "PyTorch and NumPy neighborhood implementations give different results ❌"
        )

    return


if __name__ == "__main__":

    ######################################################
    # Get MACE model
    ######################################################
    # mace_off or mace_anicc
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    dtypestr = "float32"
    use_cueq = False
    batch_size = 10
    calc = mace_off(
        model="small", device=device_str, dtype=dtypestr, enable_cueq=use_cueq
    )
    # calc = mace_anicc(device=device_str, dtype=dtypestr, enable_cueq=True)
    device = calc.device

    ######################################################
    # get alanine dipeptide atoms
    ######################################################
    atoms = load_alanine_dipeptide_ase()
    atoms.calc = calc
    atoms_calc = atoms.calc

    # ASE atoms -> torch batch
    batch_base = atoms_calc._atoms_to_batch(copy.deepcopy(atoms))

    if atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
        batch = atoms_calc._clone_batch(batch_base)
        node_heads = batch["head"][batch["batch"]]
        num_atoms_arange = torch.arange(batch["positions"].shape[0])
        node_e0 = atoms_calc.models[0].atomic_energies_fn(batch["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        compute_stress = not atoms_calc.use_compile
    else:
        compute_stress = False
    batch_base = batch_base.to_dict()

    model = atoms_calc.models[0]
    atoms_calc = atoms_calc

    # Make minibatch version of batch, that is just multiple copies of the same AlDi configuration
    # but mimicks a batch from a typical torch_geometric dataloader
    # and that we can then rotate to get different phi/psi values
    # if use_vmap:
    # batch_size of one that is duplicated by vmap
    singlebatch_base = repeated_atoms_to_batch(
        atoms_calc, copy.deepcopy(atoms), bs=1, repeats=1
    )

    minibatch_base = repeated_atoms_to_batch(
        atoms_calc,
        copy.deepcopy(atoms),
        bs=batch_size,
        repeats=batch_size,
    )

    ######################################################
    # Run tests
    ######################################################
    test_if_energy_depends_r_max(model, singlebatch_base)  # ✅ does not depend on r_max
    test_if_energy_depends_cell(model, singlebatch_base)  # ✅ does not depend on cell
    test_torch_vs_numpy_neighborhood(model, singlebatch_base)  # ✅
