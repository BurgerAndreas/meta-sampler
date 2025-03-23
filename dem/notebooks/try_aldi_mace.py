from typing import List, Dict, Tuple, Optional

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

import time

import torch_geometric as tg

from alanine_dipeptide.mace_neighbourhood import (
    update_neighborhood_graph_batched,
    update_neighborhood_graph_torch,
    update_neighborhood_graph_torch_batched,
    get_neighborhood,
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

from dem.models.components.score_estimator import estimate_grad_Rt

from dem.energies.alanine_dipeptide_energy import (
    VectorizedMACE,
    compute_hessians_vmap,
    tensor_like,
)


######################################################
# Get MACE model
######################################################
# mace_off or mace_anicc
device_str = "cuda" if torch.cuda.is_available() else "cpu"
dtypestr = "float32"
use_cueq = False
batch_size = 10
calc = mace_off(model="small", device=device_str, dtype=dtypestr, enable_cueq=use_cueq)
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
vectorized_model = VectorizedMACE(model)

for interaction, product, readout in zip(
    model.interactions, model.products, model.readouts
):
    print(interaction.__class__.__name__)
    print(product.__class__.__name__)
    print(readout.__class__.__name__)

# Make minibatch version of batch, that is just multiple copies of the same AlDi configuration
# but mimicks a batch from a typical torch_geometric dataloader
# and that we can then rotate to get different phi/psi values
# if use_vmap:
# batch_size of one that is duplicated by vmap
singlebatch_base = repeated_atoms_to_batch(
    atoms_calc, copy.deepcopy(atoms), bs=1, repeats=1
)
# totally connected graph so that we don't have to update the edge indices for every dihedral angle
singlebatch_base = update_neighborhood_graph_torch(
    singlebatch_base,
    model.r_max.item() * 1000,
)

minibatch_base = repeated_atoms_to_batch(
    atoms_calc,
    copy.deepcopy(atoms),
    bs=batch_size,
    repeats=batch_size,
)
# totally connected graph so that we don't have to update the edge indices for every dihedral angle
minibatch_base = update_neighborhood_graph_torch_batched(
    minibatch_base,
    model.r_max.item() * 1000,
)


####################################################################################################
# Energy forward pass (without forces and hessian) of MACE
####################################################################################################


@torch.jit.unused
def _energy_vmap(
    samples: torch.Tensor, return_aux_output: bool = False
) -> torch.Tensor:
    """Takes in a samples of phi/psi values and returns the energies."""
    minibatch = atoms_calc._clone_batch(singlebatch_base)
    # minibatch = singlebatch_base.clone()
    # positions_list = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0) # [B, N_atoms, 3]

    def _dihedrals_to_energies(
        _phi_psi, positions, node_attrs, edge_index, batch, head, shifts, ptr
    ):
        # positions [N, 3]
        # phi_psi [2]
        _positions = positions
        # Update xyz positions of atoms based on phi/psi values
        _positions1 = set_dihedral_torch_vmap(
            _positions, "phi", _phi_psi[0], "phi", "bg"
        )
        positions = set_dihedral_torch_vmap(
            _positions1, "psi", _phi_psi[1], "psi", "bg"
        )

        # Update edge indices
        # not vmap-able, because the shape of edge_index is data dependent
        # minibatch = update_neighborhood_graph_torch(
        #     minibatch,
        #     model.r_max.item(),
        # )

        # Compute energies
        result = vectorized_model.forward(
            positions, node_attrs, edge_index, batch, head, shifts, ptr
        )
        return result["energy"]

    positions = minibatch["positions"]
    node_attrs = minibatch["node_attrs"]
    edge_index = minibatch["edge_index"]
    batch = minibatch["batch"]
    head = minibatch["head"][batch] if "head" in minibatch else torch.zeros_like(batch)
    shifts = minibatch["shifts"]
    ptr = minibatch["ptr"]

    # [B, 1]
    _dihedrals_to_energies_vmapped = torch.vmap(
        _dihedrals_to_energies, in_dims=(0, None, None, None, None, None, None, None)
    )
    energies = _dihedrals_to_energies_vmapped(
        samples, positions, node_attrs, edge_index, batch, head, shifts, ptr
    )
    energies = energies.squeeze(-1)

    if return_aux_output:
        aux_output = {}
        return energies, aux_output
    return energies


####################################################################################################
# Psuedo energy forward pass using forces and hessian of MACE
####################################################################################################


def try_mace_hessian():
    print("-" * 60)
    minibatch = minibatch_base.clone()
    out = model(minibatch, compute_stress=False, training=True)
    print("out['energy'].shape: ", out["energy"].shape)
    print("out['forces'].shape: ", out["forces"].shape)

    # Compute Hessian
    out = model(minibatch, compute_stress=False, compute_hessian=True, training=True)
    print("out['hessian'].shape: ", out["hessian"].shape)
    print("-" * 60)
    return True


def _pseudoenergy_vmap(
    samples: torch.Tensor, return_aux_output: bool = False
) -> torch.Tensor:
    """Takes in samples of phi/psi values and returns the pseudoenergy.
    Args:
        samples (torch.Tensor): [B, 2]
    Returns:
        torch.Tensor: [B]
    """
    minibatch = atoms_calc._clone_batch(singlebatch_base)
    # minibatch = singlebatch_base.clone()

    def _dihedrals_to_energies(
        _phi_psi, positions, node_attrs, edge_index, batch, head, shifts, ptr
    ):
        # Update xyz positions of atoms based on phi/psi values
        positions1 = set_dihedral_torch_vmap(positions, "phi", _phi_psi[0], "phi", "bg")
        positions2 = set_dihedral_torch_vmap(
            positions1, "psi", _phi_psi[1], "psi", "bg"
        )

        # Update edge indices
        # not vmap-able, because the shape of edge_index is data dependent
        # minibatch = update_neighborhood_graph_torch(
        #     minibatch,
        #     model.r_max.item(),
        # )

        result = vectorized_model.forward(
            positions2, node_attrs, edge_index, batch, head, shifts, ptr
        )
        return result["energy"].squeeze(-1)

    # Get input variables as tensors
    positions = minibatch["positions"]
    node_attrs = minibatch["node_attrs"]
    edge_index = minibatch["edge_index"]
    batch = minibatch["batch"]
    head = minibatch["head"][batch] if "head" in minibatch else torch.zeros_like(batch)
    shifts = minibatch["shifts"]
    ptr = minibatch["ptr"]

    # [B]
    _dihedrals_to_energies_vmapped = torch.vmap(
        _dihedrals_to_energies, in_dims=(0, None, None, None, None, None, None, None)
    )
    energies = _dihedrals_to_energies_vmapped(
        samples, positions, node_attrs, edge_index, batch, head, shifts, ptr
    )

    # forces = gradient of energy with respect to phi/psi [B, 2]
    # forces = -torch.func.grad(_dihedrals_to_energies_vmapped, argnums=0)(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)
    forces = -1 * torch.vmap(
        torch.func.grad(_dihedrals_to_energies, argnums=0),
        in_dims=(0, None, None, None, None, None, None, None),
    )(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)

    # Force magnitude [B]
    forces_norm = torch.linalg.norm(forces, dim=1)  # [B]

    # compute Hessian [B, 2, 2]
    # hessian = torch.func.hessian(_dihedrals_to_energies_vmapped, argnums=0)(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)
    hessian = torch.vmap(
        torch.func.hessian(_dihedrals_to_energies, argnums=0),
        in_dims=(0, None, None, None, None, None, None, None),
    )(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)

    # [B, D], [B, D, D]
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)

    # [B]
    eigval_product = eigenvalues[:, 0] * eigenvalues[:, 1]

    # Pseudoenergy [B]
    energy_loss = 1.0 * energies
    force_loss = 1.0 * forces_norm
    hessian_loss = 1.0 * eigval_product
    total_loss = energy_loss + force_loss + hessian_loss

    if return_aux_output:
        aux_output = {}
        return total_loss, aux_output
    return total_loss


# TODO: Dummy that does not use input
def _pseudoenergy_vmap_allatoms(
    samples: torch.Tensor, return_aux_output: bool = False
) -> torch.Tensor:
    """Takes in samples of phi/psi values and returns the pseudoenergy.
    Args:
        samples (torch.Tensor): [B, N]
    Returns:
        torch.Tensor: [B]
    """
    minibatch = atoms_calc._clone_batch(singlebatch_base)
    # minibatch = singlebatch_base.clone()

    def _get_energy(positions, node_attrs, edge_index, batch, head, shifts, ptr):
        # Update xyz positions of atoms based on phi/psi values
        # positions1 = set_dihedral_torch_vmap(
        #     positions, "phi", _phi_psi[0], "phi", 'bg'
        # )
        # positions2 = set_dihedral_torch_vmap(
        #     positions1, "psi", _phi_psi[1], "psi", 'bg'
        # )

        # Update edge indices
        # not vmap-able, because the shape of edge_index is data dependent
        # minibatch = update_neighborhood_graph_torch(
        #     minibatch,
        #     model.r_max.item(),
        # )

        result = vectorized_model(
            positions, node_attrs, edge_index, batch, head, shifts, ptr
        )
        return result["energy"].squeeze(-1)

    # Get input variables as tensors
    positions = minibatch["positions"]
    node_attrs = minibatch["node_attrs"]
    edge_index = minibatch["edge_index"]
    batch = minibatch["batch"]
    head = minibatch["head"][batch] if "head" in minibatch else torch.zeros_like(batch)
    shifts = minibatch["shifts"]
    ptr = minibatch["ptr"]

    # repeat positions B times
    positions = positions.repeat(batch.shape[0], 1, 1)

    # TODO: Hessian causes issues
    # compute Hessian [B, 2, 2]
    # hessian = torch.func.hessian(_get_energy, argnums=0)(samples, positions, node_attrs, edge_index, batch, head, shifts, ptr)
    hessian = torch.vmap(
        torch.func.hessian(_get_energy, argnums=0),
        in_dims=(0, None, None, None, None, None, None),
    )(positions, node_attrs, edge_index, batch, head, shifts, ptr)
    print("hessian.shape: ", hessian.shape)

    # Pseudoenergy [B]
    hessian_loss = 1.0 * torch.mean(hessian, dim=(1, 2))
    total_loss = hessian_loss
    print("total_loss.shape: ", total_loss.shape)

    if return_aux_output:
        aux_output = {}
        return total_loss, aux_output
    return total_loss


# TODO: figure out Hessian computation with batching
def _pseudoenergy_batched(samples: torch.Tensor) -> torch.Tensor:
    bs = samples.shape[0]
    if bs == batch_size:
        minibatch = minibatch_base.clone()
    else:
        # construct a minibatch with the correct batch size
        minibatch = repeated_atoms_to_batch(
            atoms_calc, copy.deepcopy(atoms), bs=bs, repeats=bs
        )
    minibatch = atoms_calc._clone_batch(minibatch)

    # Update xyz positions of atoms based on phi/psi values
    # forces = gradient of energy with respect to phi/psi
    phi_psi_batch = samples.requires_grad_(True)
    minibatch = update_alanine_dipeptide_xyz_from_dihedrals_batched(
        phi_psi_batch, minibatch, convention="bg"
    )

    # Update edge indices
    # TODO: why is the cell so huge?
    minibatch = update_neighborhood_graph_batched(
        minibatch, model.r_max.item(), overwrite_cell=True
    )

    # Compute energies
    out = model(minibatch, compute_stress=False, training=True)

    # Compute forces [B, 2]
    forces = torch.autograd.grad(
        outputs=out["energy"],  # [B]
        inputs=phi_psi_batch,  # [B, 2]
        grad_outputs=torch.ones_like(out["energy"], device=device),  # [B]
        create_graph=True,
        retain_graph=True,
    )[0]
    forces_norm = torch.linalg.norm(forces, dim=1)

    # Compute Hessian [B, 2, 2]

    # Hessian v1: not working
    # hessian = torch.autograd.functional.hessian(
    #     func=lambda x: out["energy"],
    #     inputs=phi_psi_batch,
    #     create_graph=True,
    # )

    # Hessian v2: not working
    # hessian = torch.func.hessian(get_energy)(samples)
    #     eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    #     # Sort eigenvalues and corresponding eigenvectors
    #     sorted_indices = torch.argsort(eigenvalues)
    #     eigenvalues = eigenvalues[sorted_indices]
    #     eigenvectors = eigenvectors[:, sorted_indices]
    #     # Get 2 smallest eigenvalues and their eigenvectors
    #     smallest_eigenvalues = eigenvalues[:2]
    #     smallest_eigenvectors = eigenvectors[:, :2]

    # TODO: Hessian should be [B, d, d]
    print("forces.shape: ", forces.shape)
    print("phi_psi_batch.shape: ", phi_psi_batch.shape)
    hessian = compute_hessians_vmap(forces, phi_psi_batch)  # [B*d, B, d]
    print("hessian.shape: ", hessian.shape)

    # TODO: computing Hessian for each sample results in no gradients?
    print("forces[0].shape: ", forces[0].shape)
    print("phi_psi_batch[0].shape: ", phi_psi_batch[0].shape)
    hessian = compute_hessians_vmap(forces[0], phi_psi_batch[0])
    print("hessian.shape: ", hessian.shape)

    # compute smallest eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    smallest_eigenvalues = eigenvalues[:2]
    smallest_eigenvectors = eigenvectors[:, :2]
    eigval_product = smallest_eigenvalues[0] * smallest_eigenvalues[1]

    pseudoenergy = 1.0 * out["energy"] + 1.0 * forces_norm + 1.0 * eigval_product
    return pseudoenergy


def _pseudoenergy_batched_loop(
    samples: torch.Tensor, return_aux_output: bool = False
) -> torch.Tensor:
    """
    Compute pseudoenergy for a batch of dihedral angles
    Args:
        samples (torch.Tensor): [B, 2]
    Returns:
        torch.Tensor: [B]
    """
    bs = samples.shape[0]

    pseudoenergies = []
    for sample in samples:
        minibatch = singlebatch_base.clone()
        minibatch = atoms_calc._clone_batch(minibatch)

        # Update xyz positions of atoms based on phi/psi values
        # forces = gradient of energy with respect to phi/psi
        phi_psi = sample.requires_grad_(True)
        positions = minibatch["positions"]  # [N, 3]
        positions1 = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", "bg")
        minibatch["positions"] = set_dihedral_torch(
            positions1, "psi", phi_psi[1], "psi", "bg"
        )

        # Update edge indices
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=minibatch["positions"].detach().cpu().numpy(),
            cutoff=model.r_max.item(),
            cell=minibatch["cell"].detach().cpu().numpy(),  # [3, 3]
        )
        minibatch["edge_index"] = tensor_like(edge_index, minibatch["edge_index"])
        minibatch["shifts"] = tensor_like(shifts, minibatch["shifts"])
        minibatch["unit_shifts"] = tensor_like(unit_shifts, minibatch["unit_shifts"])
        minibatch["cell"] = tensor_like(cell, minibatch["cell"])

        # Compute energies
        out = model(minibatch, compute_stress=compute_stress, training=True)

        # Compute forces [2]
        forces = torch.autograd.grad(
            outputs=out["energy"],  # [1]
            inputs=phi_psi,  # [2]
            grad_outputs=torch.ones_like(out["energy"], device=device),  # [2]
            create_graph=True,
            retain_graph=True,
        )[0]
        forces_norm = torch.linalg.norm(forces)

        hessian = compute_hessians_vmap(forces, phi_psi)  # [d, d]

        # compute smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        smallest_eigenvalues = eigenvalues[:2]
        smallest_eigenvectors = eigenvectors[:, :2]
        eigval_product = smallest_eigenvalues[0] * smallest_eigenvalues[1]

        pseudoenergy = 1.0 * out["energy"] + 1.0 * forces_norm + 1.0 * eigval_product
        pseudoenergies.append(pseudoenergy)

    pseudoenergies = torch.stack(pseudoenergies, dim=0).squeeze(1)
    if return_aux_output:
        aux_output = {}
        return pseudoenergies, aux_output
    return pseudoenergies


####################################################################################################
# Tests
####################################################################################################


# Energy + force + hessian, all atoms
def test_score_estimator_pseudoenergy_allatoms_vmap():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input."""
    # from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    print("")
    print("=" * 80)
    print("Test score estimator pseudoenergy_allatoms_vmap")

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2  # 4
    dim = 2
    num_mc_samples = 2  # 8

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # energy_function = DoubleWellEnergy(device=device, dimensionality=dim, use_vmap=True)
    energy_function = _pseudoenergy_vmap_allatoms

    print(f"Running with vmap=True, batch_size={batch_size}, dim={dim}")
    grad_output, aux_output = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=True,
        return_aux_output=True,
    )
    print(f"Gradient shape: {grad_output.shape}")
    print(
        f"Gradient mean: {grad_output.mean().item():.4f}, std: {grad_output.std().item():.4f}"
    )
    print(f"Auxiliary output keys: {aux_output.keys()}")
    print("_pseudoenergy_vmap_allatoms ✅")
    return True


# Energy + force + hessian
def test_score_estimator_pseudoenergy_vmap():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input."""
    # from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    print("")
    print("=" * 80)
    print("Test score estimator pseudoenergy_vmap")

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    dim = 2
    num_mc_samples = 4

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # energy_function = DoubleWellEnergy(device=device, dimensionality=dim, use_vmap=True)
    energy_function = _pseudoenergy_vmap

    t1 = time.time()
    grad_output, aux_output = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=True,
        return_aux_output=True,
    )
    t2 = time.time()
    print(f"Time gradient estimation vmap: {t2 - t1:.4f} seconds")
    print(f"Gradient shape: {grad_output.shape}")
    print(f"Auxiliary output keys: {aux_output.keys()}")
    print("_pseudoenergy_batched_loop ✅")
    return True


# Energy + force + hessian
def test_score_estimator_pseudoenergy_batched_loop():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input.
    Original DEM codebase uses vmap and torch.func.grad, we use a batched version instead
    """
    print("")
    print("=" * 80)
    print("Test score estimator pseudoenergy_batched_loop")
    # from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4  # num_samples_to_sample_from_buffer
    dim = 2
    num_mc_samples = 16  # num_estimator_mc_samples

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # Test with vmap=False
    print(f"Running with vmap=False, batch_size={batch_size}, dim={dim}")
    # energy_function = DoubleWellEnergy(device=device, dimensionality=dim, use_vmap=False)
    energy_function = _pseudoenergy_batched_loop
    energy_output = energy_function(x)
    print(f"Energy shape: {energy_output.shape}")
    assert energy_output.shape == (
        batch_size,
    ), f"Energy shape should be (batch_size,), got {energy_output.shape}"

    print("Estimating score...")
    t1 = time.time()
    grad_output_no_vmap, aux_output_no_vmap = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=False,
        return_aux_output=True,
    )
    t2 = time.time()
    print(f"Time gradient estimation no vmap: {t2 - t1:.4f} seconds")
    print(f"Gradient shape: {grad_output_no_vmap.shape}")
    print(f"Auxiliary output keys: {aux_output_no_vmap.keys()}")
    assert grad_output_no_vmap.shape == (
        batch_size,
        dim,
    ), f"Gradient shape should be (batch_size, dim), got {grad_output_no_vmap.shape}"
    print("_pseudoenergy_batched_loop ✅")
    return True


# Energy only (no forces or hessians)
def test_score_estimator_energy_vmap():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input."""
    print("")
    print("=" * 80)
    print("Test score estimator energy_vmap")
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4  # num_samples_to_sample_from_buffer
    dim = 2
    num_mc_samples = 16  # num_estimator_mc_samples

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # Test with vmap=True
    energy_function = _energy_vmap
    print(f"Running with vmap=True, batch_size={batch_size}, dim={dim}")
    energy_output = energy_function(x)
    print(f"Energy shape: {energy_output.shape}")
    assert energy_output.shape == (
        batch_size,
    ), f"Energy shape should be (batch_size,), got {energy_output.shape}"
    print(
        f"Energy mean: {energy_output.mean().item():.4f}, std: {energy_output.std().item():.4f}"
    )

    print("Estimating score...")
    t1 = time.time()
    grad_output, aux_output = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=True,
        return_aux_output=True,
    )
    t2 = time.time()
    print(f"Time gradient estimation vmap: {t2 - t1:.4f} seconds")
    print(f"Gradient shape: {grad_output.shape}")
    print(f"Auxiliary output keys: {aux_output.keys()}")
    assert grad_output.shape == (
        batch_size,
        dim,
    ), f"Gradient shape should be (batch_size, dim), got {grad_output.shape}"
    print("_energy_vmap ✅")


def test_forward_pass_pseudoenergy_batched_loop():
    # Create test dihedral angles
    # Create a grid of phi/psi angles
    phi_range = torch.linspace(-np.pi, np.pi, 5)
    psi_range = torch.linspace(-np.pi, np.pi, 5)
    phi_grid, psi_grid = torch.meshgrid(phi_range, psi_range, indexing="ij")
    dihedrals = torch.stack([phi_grid.flatten(), psi_grid.flatten()], dim=1).to(device)

    print(f"Testing with {len(dihedrals)} dihedral angle pairs")

    # test with loop
    print("Testing loop version...")
    print("dihedrals.shape: ", dihedrals.shape)
    t1 = time.time()
    pseudoenergies = _pseudoenergy_batched_loop(dihedrals)
    t2 = time.time()
    print(f"Time forward pass: {t2 - t1:.4f} seconds")
    print(f"Pseudoenergies shape: {pseudoenergies.shape}")
    print("passed! ✅")


if __name__ == "__main__":
    print("=" * 80)
    # set_jit_enabled(False)

    torch.autograd.set_detect_anomaly(True)

    # try_mace_hessian()

    ##################################################################################
    # Test score estimator (gradient of energy w.r.t. input) of AlDi energy function
    ##################################################################################

    test_score_estimator_energy_vmap()

    ##################################################################################
    # Test score estimator (gradient of energy w.r.t. input) of AlDi pseudo-energy function
    ##################################################################################

    # test_forward_pass_pseudoenergy_batched_loop()
    test_score_estimator_pseudoenergy_vmap()
    test_score_estimator_pseudoenergy_batched_loop()
