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


#############################################################################
# TODO: Vectorize MACE
#############################################################################


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

class VectorizedMACE(torch.nn.Module):
    """Wrapper around MACE model to vmap the forward pass.
    Implements force and hessian, but not virial or stress.
    """
    def __init__(self, model: mace.modules.models.MACE):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        # TODO: torch.func.grad instead of torch.autograd.grad
        data["node_attrs"].requires_grad_(True) # TODO: necessary? do outside forward?
        data["positions"].requires_grad_(True)
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        num_graphs = data["ptr"].numel() - 1
        node_heads = (
            data["head"][data["batch"]]
            if "head" in data
            else torch.zeros_like(data["batch"])
        )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        # Don't need this for our systems
        # if compute_virials or compute_stress or compute_displacement:
        #     (
        #         data["positions"],
        #         data["shifts"],
        #         displacement,
        #     ) = get_symmetric_displacement(
        #         positions=data["positions"],
        #         unit_shifts=data["unit_shifts"],
        #         cell=data["cell"],
        #         edge_index=data["edge_index"],
        #         num_graphs=num_graphs,
        #         batch=data["batch"],
        #     )

        # Atomic energies
        node_e0 = self.model.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.model.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats = self.model.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.model.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.model.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.model.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.model.interactions, self.model.products, self.model.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        # TODO: torch.func.grad instead of torch.autograd.grad
        energy=total_energy
        positions=data["positions"]
        cell=data["cell"]
        virials, stress = None, None
        if compute_force:
            forces = torch.autograd.grad(
                outputs=[energy],  # [n_graphs, ]
                inputs=[positions],  # [n_nodes, 3]
                grad_outputs=[torch.ones_like(energy)],
                retain_graph=True,  # Make sure the graph is not destroyed during training
                create_graph=True,  # Create graph for second derivative
                # allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]
            if forces is None:
                forces = torch.zeros_like(positions)
            forces = -1 * forces
        else:
            forces = None
        if compute_hessian:
            assert forces is not None, "Forces must be computed to get the hessian"
            hessian = compute_hessians_vmap(forces, positions)
        else:
            hessian = None

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
        }

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


# MACE Hessians
# https://github.com/ACEsuit/mace/blob/1d5b6a0bdfdc7258e0bb711eda1c998a4aa77976/mace/modules/utils.py#L112
# https://mace-docs.readthedocs.io/en/latest/guide/hessian.html
@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
    verbose: bool = False,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0]
    except RuntimeError:
        if verbose:
            print("RuntimeError: compute_hessians_vmap. Using compute_hessians_loop")
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


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


####################################################################################################


def tensor_like(_new, _base):
    return torch.tensor(_new, device=_base.device, dtype=_base.dtype)


def _forward_vmap(samples: torch.Tensor) -> torch.Tensor:
    """Takes in a single sample and returns the pseudoenergy.
    No batch dimension
    """
    raise NotImplementedError
    minibatch = singlebatch_base.clone()

    # Update xyz positions of atoms based on phi/psi values
    # forces = gradient of energy with respect to phi/psi
    phi_psi_batch = samples

    # Define a function to compute energy for functorch.grad
    def compute_energy(phi_psi, minibatch):
        positions = minibatch["positions"]
        print("vmap positions.shape: ", positions.shape)
        positions1 = set_dihedral_torch_vmap(
            positions, "phi", phi_psi[0], "phi", convention
        )
        print("vmap positions1.shape: ", positions1.shape)
        minibatch["positions"] = set_dihedral_torch_vmap(
            positions1, "psi", phi_psi[1], "psi", convention
        )
        print("vmap minibatch['positions'].shape: ", minibatch["positions"].shape)

        # Update edge indices
        # not vmap-able, because the shape of edge_index is data dependent
        # minibatch = update_neighborhood_graph_torch(
        #     minibatch,
        #     model.r_max.item(),
        # )

        # Compute energies
        # Mace is not vmap-able, because it uses torch grad to compute the forces
        # mace/modules/models.py", line 353, in forward
        # data["positions"].requires_grad_(True)
        # RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs)
        # inside of a function being transformed by a functorch transform.
        # This is unsupported, please attempt to use the functorch transforms (e.g. grad, vjp, jacrev, jacfwd, hessian)
        # or call requires_grad_() outside of a function being transformed instead
        result = model(minibatch, compute_stress=False, training=True)
        return result["energy"]

    # Compute energy and forces
    energy_fn = lambda x: compute_energy(x, minibatch)
    grad_fn = torch.func.grad(energy_fn)

    # Get energy and forces
    out = {"energy": compute_energy(phi_psi_batch, minibatch)}
    forces = -grad_fn(phi_psi_batch)  # Negative gradient gives forces

    forces_norm = torch.linalg.norm(forces, dim=1)

    energy_loss = (energy_weight * out["energy"]).mean()
    force_loss = (force_weight * forces_norm).mean()
    total_loss = energy_loss + force_loss
    if return_aux_output:
        aux_output = {
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "total_loss": total_loss,
        }
        return total_loss, aux_output
    return total_loss


# TODO: figure out Hessian computation with batching
def _forward_batched(samples: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
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
        phi_psi_batch, minibatch, convention=convention
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

    pseudoenergy = (
        energy_weight * out["energy"] + force_weight * forces_norm + eigval_product
    )
    return pseudoenergy


def _forward_batched_loop(samples: torch.Tensor, return_aux_output: bool = False) -> torch.Tensor:
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
        positions1 = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
        minibatch["positions"] = set_dihedral_torch(
            positions1, "psi", phi_psi[1], "psi", convention
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

        pseudoenergy = (
            energy_weight * out["energy"] + force_weight * forces_norm + eigval_product
        )
        pseudoenergies.append(pseudoenergy)

    pseudoenergies = torch.stack(pseudoenergies, dim=0).squeeze(1)
    if return_aux_output:
        aux_output = {}
        return pseudoenergies, aux_output
    return pseudoenergies


####################################################################################################
# Tests
####################################################################################################


def test_score_estimator_vmap():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input."""
    from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    dim = 2
    num_mc_samples = 100

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # Test with vmap=True
    # energy_function = DoubleWellEnergy(device=device, dimensionality=dim, use_vmap=True)
    energy_function = _forward_vmap
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
    return


def test_score_estimator_batched():
    """In DEM the energy function is called in the score estimator, which computes the gradient of the energy w.r.t. the input.
    Original DEM codebase uses vmap and torch.func.grad, we use a batched version instead
    """
    from dem.energies.double_well_energy import DoubleWellEnergy
    from dem.models.components.noise_schedules import GeometricNoiseSchedule

    # Set up test parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    dim = 2
    num_mc_samples = 16

    # Create noise schedule
    noise_schedule = GeometricNoiseSchedule(sigma_min=0.00001, sigma_max=1.0)

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # mid-point in time
    x = torch.randn(batch_size, dim, device=device)  # random positions

    # Test with vmap=False
    print(f"\nRunning with vmap=False, batch_size={batch_size}, dim={dim}")
    # energy_function = DoubleWellEnergy(device=device, dimensionality=dim, use_vmap=False)
    energy_function = _forward_batched_loop
    grad_output_no_vmap, aux_output_no_vmap = estimate_grad_Rt(
        t=t,
        x=x,
        energy_function=energy_function,
        noise_schedule=noise_schedule,
        num_mc_samples=num_mc_samples,
        use_vmap=False,
        return_aux_output=True,
    )
    print(f"Gradient shape: {grad_output_no_vmap.shape}")
    print(
        f"Gradient mean: {grad_output_no_vmap.mean().item():.4f}, std: {grad_output_no_vmap.std().item():.4f}"
    )
    print(f"Auxiliary output keys: {aux_output_no_vmap.keys()}")


if __name__ == "__main__":

    # try_mace_hessian()

    # Define parameters for testing
    batch_size = 10
    energy_weight = 1.0
    force_weight = 0.1
    convention = "bg"
    return_aux_output = False

    # Create test dihedral angles
    # Create a grid of phi/psi angles
    phi_range = torch.linspace(-np.pi, np.pi, 5)
    psi_range = torch.linspace(-np.pi, np.pi, 5)
    phi_grid, psi_grid = torch.meshgrid(phi_range, psi_range, indexing="ij")
    dihedrals = torch.stack([phi_grid.flatten(), psi_grid.flatten()], dim=1).to(device)

    print(f"Testing with {len(dihedrals)} dihedral angle pairs")
    print("=" * 80)

    ##################################################################################
    # Test forward passes of AlDi energy function
    ##################################################################################

    # test with loop
    print("Testing loop version...")
    print("dihedrals.shape: ", dihedrals.shape)
    t1 = time.time()
    pseudoenergies = _forward_batched_loop(dihedrals)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.4f} seconds")
    print(f"Pseudoenergies shape: {pseudoenergies.shape}")
    print("passed! ✅")

    # # Test without vmap (batched version)
    # print("Testing batched version...")
    # batch_energies = _forward_batched(dihedrals[:batch_size])
    # print(f"Batch energies shape: {batch_energies.shape}")

    # # Test with vmap
    # print("\nTesting vmap version...")
    # # Use torch.vmap to vectorize the _forward_vmap function over the batch dimension
    # t1 = time.time()
    # vmapped_forward = torch.vmap(_forward_vmap)
    # t2 = time.time()
    # print(f"Time taken: {t2 - t1:.4f} seconds")
    # # Apply the vmapped function to the first batch_size samples
    # vmap_energies = vmapped_forward(dihedrals[: min(batch_size, len(dihedrals))])
    # print(f"Vmap energies shape: {vmap_energies.shape}")
    # print("passed! ✅")

    ##################################################################################
    # Test score estimator (gradient of energy w.r.t. input) of AlDi energy function
    ##################################################################################

    test_score_estimator_batched()
    # test_score_estimator_vmap()
