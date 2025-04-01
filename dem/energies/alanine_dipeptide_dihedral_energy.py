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

# from mace.tools import torch_geometric, torch_tools, utils
import mace.data
import ase
import ase.io
import ase.build

# from ase.calculators.calculator import Calculator, all_changes
# import openmm
# from openmm.unit import nanometer
# from openmm.unit import Quantity

# import torch_geometric as tg
from tqdm import tqdm
import itertools
import pickle
import os


from dem.utils.plotting import get_logp_on_grid

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


def tensor_like(_new, _base):
    return torch.tensor(_new, device=_base.device, dtype=_base.dtype)


def set_jit_enabled(enabled: bool):
    """Enables/disables JIT"""
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()


def jit_enabled():
    """Returns whether JIT is enabled"""
    if torch.__version__ < "1.7":
        return torch.jit._enabled
    else:
        return torch.jit._state._enabled.enabled


#############################################################################
# Vectorize MACE
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
    # [n_edges, 3]
    vectors = positions[receiver] - positions[sender] + shifts
    # For some reason, torch.linalg.norm throws an 'inplace modification' error with torch.func.hessian
    # lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    lengths = torch.sum(vectors**2, dim=-1, keepdim=True)
    lengths = lengths.sqrt()
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths
    return vectors, lengths


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
        # self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
        # return out.scatter_add_(dim, index, src)
        return torch.scatter_add(out, dim, index, src)
    else:
        # return out.scatter_add_(dim, index, src)
        return torch.scatter_add(out, dim, index, src)


class VectorizedMACE(torch.nn.Module):
    """Wrapper around MACE model to vmap the forward pass.
    Implements force and hessian, but not virial or stress.
    """

    def __init__(self, model: mace.modules.models.MACE):
        super().__init__()
        self.model = model

    # @torch.jit.unused
    def forward(
        self,
        positions: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        shifts: torch.Tensor,
        ptr: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Returns only energy. Virials, stress, etc. not implemented."""

        num_atoms_arange = torch.arange(positions.shape[0])
        num_graphs = ptr.numel() - 1
        node_heads = head[batch]
        # node_heads = (
        #     data["head"][data["batch"]]
        #     if "head" in data
        #     else torch.zeros_like(data["batch"])
        # )

        # Atomic energies
        node_e0 = self.model.atomic_energies_fn(node_attrs)[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=batch, dim=0, dim_size=num_graphs
        )  # [n_graphs, n_heads]

        # Embeddings
        node_feats = self.model.node_embedding(node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions.clone(),
            edge_index=edge_index.clone(),
            shifts=shifts.clone(),
        )
        edge_attrs = self.model.spherical_harmonics(vectors.clone())
        edge_feats = self.model.radial_embedding(
            lengths.clone(), node_attrs, edge_index, self.model.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.model.pair_repulsion_fn(
                lengths.clone(),
                node_attrs,
                edge_index,
                self.model.atomic_numbers,
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=batch, dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        # Interactions
        energies = [e0, pair_energy]
        # node_energies_list = [node_e0, pair_node_energy]
        # node_feats_list = []
        for interaction, product, readout in zip(
            self.model.interactions, self.model.products, self.model.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            # node_feats_list.append(node_feats)
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=batch,
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            # node_energies_list.append(node_energies)

        # Concatenate node features
        # node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        # node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        # node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        return {
            "energy": total_energy,
            # "node_energy": node_energy,
            # "contributions": contributions,
            # "node_feats": node_feats_out,
        }

    # TODO: shape error? maceoff is ScaleShiftMACE?
    # @torch.jit.unused
    def forward_ScaleShiftMACE(
        self,
        positions: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        shifts: torch.Tensor,
        ptr: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Returns only energy. Virials, stress, etc. not implemented."""

        num_atoms_arange = torch.arange(positions.shape[0])
        num_graphs = ptr.numel() - 1
        node_heads = head[batch]
        # node_heads = (
        #     data["head"][data["batch"]]
        #     if "head" in data
        #     else torch.zeros_like(data["batch"])
        # )

        # Atomic energies
        node_e0 = self.model.atomic_energies_fn(node_attrs)[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=batch, dim=0, dim_size=num_graphs
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.model.node_embedding(node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats = self.model.radial_embedding(
            lengths, node_attrs, edge_index, self.model.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.model.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.model.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.model.interactions, self.model.products, self.model.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            node_feats_list.append(node_feats)
            node_es_list.append(
                readout(node_feats, node_heads)[num_atoms_arange, node_heads]
            )  # {[n_nodes, ], }

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.model.scale_shift(node_inter_es, node_heads)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=batch, dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        # forces are computed from inter_e?
        return {
            "energy": total_energy,
            # "node_energy": node_energy,
            # "contributions": contributions,
            # "node_feats": node_feats_out,
        }


class MaceAlDiEnergy2D(BaseEnergyFunction):
    """
    Energy function for alanine dipeptide using dihedral angles as collective variables.
    """

    def __init__(
        self,
        dimensionality=2,
        device="cuda",
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=2.0,  # ?
        convention="bg",
        use_vmap=True,
        plotting_batch_size=128,
        # loss weights
        force_weight=1.0,
        energy_weight=0.1,
        # Mace
        use_cueq=True,
        dtypestr="float32",
        batch_size=1,
        use_scale_shift=True,
        # shift zero point of energy so that energies are positive 
        # so that exponentials are nice and finite
        shift_energy=13494.408,  # 13494.406879236485
        #
        *args,
        **kwargs,
    ):
        torch.manual_seed(0)
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.shift_energy = shift_energy

        self.curr_epoch = 0

        # self.energy_weight = energy_weight
        # self.force_weight = force_weight
        self.batch_size = batch_size
        self.use_vmap = use_vmap
        self.convention = convention
        ######################################################
        # Get MACE model
        ######################################################
        # mace_off or mace_anicc
        # device_str = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(
            model="small", device=device, dtype=dtypestr, enable_cueq=use_cueq
        )
        # calc = mace_anicc(device=device_str, dtype=dtypestr, enable_cueq=True)

        ######################################################
        # get alanine dipeptide atoms
        ######################################################
        atoms = load_alanine_dipeptide_ase()
        atoms.calc = calc
        atoms_calc = atoms.calc

        # ASE atoms -> torch batch
        batch_base = atoms_calc._atoms_to_batch(copy.deepcopy(atoms))

        self.batch_base = batch_base.to_dict()
        self.atoms_calc = atoms_calc
        self.atoms = atoms
        self.model = atoms_calc.models[0]
        print(self.__class__.__name__, "model type:", self.model.__class__.__name__)
        self.vectorized_model = VectorizedMACE(self.model)

        # Make minibatch version of batch, that is just multiple copies of the same AlDi configuration
        # but mimicks a batch from a typical torch_geometric dataloader
        # and that we can then rotate to get different phi/psi values
        if self.use_vmap:
            # batch_size of one that is duplicated by vmap
            self.singlebatch_base = repeated_atoms_to_batch(
                self.atoms_calc, copy.deepcopy(self.atoms), bs=1, repeats=1
            )
            # totally connected graph so that we don't have to update the edge indices for every dihedral angle
            self.singlebatch_base = update_neighborhood_graph_torch(
                self.singlebatch_base,
                self.model.r_max.item() * 1000,
            )
        else:
            self.minibatch = repeated_atoms_to_batch(
                self.atoms_calc,
                copy.deepcopy(self.atoms),
                bs=batch_size,
                repeats=batch_size,
            )
            # totally connected graph so that we don't have to update the edge indices for every dihedral angle
            self.minibatch_base = update_neighborhood_graph_torch_batched(
                self.minibatch_base,
                self.model.r_max.item() * 1000,
            )

        ######################################################
        # Initialize BaseEnergyFunction
        ######################################################
        super().__init__(
            dimensionality=2,
            is_molecule=False,  # TODO: change this
            # normalization for the force?
            # normalization_min=-4000,
            # normalization_max=-3000,  # TODO: change this
            plotting_batch_size=plotting_batch_size,
            plotting_bounds=(-np.pi, np.pi),
            *args,
            **kwargs,
        )
        # Change the name
        self.name = self.__class__.__name__
        self.use_scale_shift = use_scale_shift
        if use_scale_shift:
            self.name = self.name + "_ScaleShift"
        self.move_to_device(self.device)

    def move_to_device(self, device):
        self.vectorized_model.to(device)
        self.model.to(device)

    def vectorized_mace_forward(self, *args, **kwargs):
        if self.use_scale_shift:
            return self.vectorized_model.forward_ScaleShiftMACE(*args, **kwargs)
        else:
            return self.vectorized_model.forward(*args, **kwargs)

    @torch.jit.unused
    def _energy_vmap(
        self,
        samples: torch.Tensor,
        return_aux_output: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Takes in a samples of phi/psi values and returns the energies."""
        minibatch = self.atoms_calc._clone_batch(self.singlebatch_base)
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
            result = self.vectorized_mace_forward(
                positions, node_attrs, edge_index, batch, head, shifts, ptr
            )
            return result["energy"]

        # Get one copy of the minibatch on the same device as the samples
        positions = minibatch["positions"].to(samples.device)
        node_attrs = minibatch["node_attrs"].to(samples.device)
        edge_index = minibatch["edge_index"].to(samples.device)
        batch = minibatch["batch"].to(samples.device)
        head = (
            minibatch["head"][batch] if "head" in minibatch else torch.zeros_like(batch)
        ).to(samples.device)
        shifts = minibatch["shifts"].to(samples.device)
        ptr = minibatch["ptr"].to(samples.device)

        # If a single sample, add a batch dimension [2] -> [1, 2]
        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)

        # [B, 1]
        _dihedrals_to_energies_vmapped = torch.vmap(
            _dihedrals_to_energies,
            in_dims=(0, None, None, None, None, None, None, None),
        )
        energies = _dihedrals_to_energies_vmapped(
            samples, positions, node_attrs, edge_index, batch, head, shifts, ptr
        )
        energies = energies.squeeze(-1)

        energies = energies + self.shift_energy
        energies = energies / temperature

        if return_aux_output:
            aux_output = {}
            return energies, aux_output
        return energies

    def _energy_batched(
        self,
        samples: torch.Tensor,
        return_aux_output: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Takes in a single sample and returns the energy.
        No batch dimension
        """
        minibatch = self.atoms_calc._clone_batch(self.singlebatch_base)
        # minibatch = self.singlebatch_base.clone()

        # Update xyz positions of atoms based on phi/psi values
        # forces = gradient of energy with respect to phi/psi
        phi_psi = samples

        positions = minibatch["positions"]
        print("batched positions.shape: ", positions.shape)
        print("batched phi_psi.shape: ", phi_psi.shape)
        positions1 = set_dihedral_torch_batched(
            positions, "phi", phi_psi[0], "phi", self.convention
        )
        print("batched positions1.shape: ", positions1.shape)
        minibatch["positions"] = set_dihedral_torch_batched(
            positions1, "psi", phi_psi[1], "psi", self.convention
        )
        print("batched minibatch['positions'].shape: ", minibatch["positions"].shape)

        # Update edge indices
        # not vmap-able, because the shape of edge_index is data dependent
        # minibatch = update_neighborhood_graph_torch(
        #     minibatch,
        #     model.r_max.item(),
        # )

        # Compute energies
        result = self.model(minibatch, training=True)
        energy = result["energy"]

        energy = energy + self.shift_energy
        energy = energy / temperature

        if return_aux_output:
            aux_output = {}
            return energy, aux_output
        return energy

    def _energy(
        self,
        samples: torch.Tensor,
        temperature: float = 1.0,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Evaluates the pseudoenergy function at given samples.

        Args:
            samples (torch.Tensor): Input points (dihedral angles)

        Returns:
            torch.Tensor: Energy values at input points
        """
        # project dihedral angles to be within [-pi, pi]
        samples = self.project_dihedral_angles(samples)
        if self.use_vmap:
            return self._energy_vmap(
                samples, return_aux_output=return_aux_output, temperature=temperature
            )
        else:
            return self._energy_batched(
                samples, return_aux_output=return_aux_output, temperature=temperature
            )

    def log_prob(
        self,
        samples: torch.Tensor,
        temperature: float = 1.0,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Evaluates the pseudoenergy function at given samples.

        Args:
            samples (torch.Tensor): Input points (dihedral angles)

        Returns:
            torch.Tensor: Energy values at input points
        """
        if return_aux_output:
            aux_output = {}
            energy, aux_output = self._energy(
                samples, return_aux_output=True, temperature=temperature
            )
            return -energy, aux_output
        return -self._energy(samples, temperature=temperature)

    def project_dihedral_angles(self, samples):
        # # Project dihedral angles to be within [-pi, pi]
        # # Handle different input shapes
        # if len(samples.shape) == 1:  # Single sample [2]
        #     if list(self._plotting_bounds) == [-np.pi, np.pi]:
        #         samples = torch.remainder(samples + torch.pi, 2 * torch.pi) - torch.pi
        #     elif list(self._plotting_bounds) == [0, 2 * np.pi]:
        #         samples = torch.remainder(samples, 2 * torch.pi)
        #     else:
        #         raise ValueError(f"Unexpected plotting bounds: {self._plotting_bounds}")
        # elif len(samples.shape) == 2:  # Batch of samples [B, 2]
        #     if list(self._plotting_bounds) == [-np.pi, np.pi]:
        #         samples = torch.remainder(samples + torch.pi, 2 * torch.pi) - torch.pi
        #     elif list(self._plotting_bounds) == [0, 2 * np.pi]:
        #         samples = torch.remainder(samples, 2 * torch.pi)
        #     else:
        #         raise ValueError(f"Unexpected plotting bounds: {self._plotting_bounds}")
        # else:
        #     raise ValueError(f"Unexpected shape for samples: {samples.shape}")

        # # # Ensure all values are within the expected range (not vmap-able)
        # # assert torch.all(samples >= -torch.pi) and torch.all(samples <= torch.pi), \
        # #     f"Samples out of range: min={samples.min()}, max={samples.max()}"
        return samples

    #####################################################################################
    # helper functions
    #####################################################################################

    def _dataset_from_minima(self, size):
        minima = self.get_minima()
        samples = minima
        if size > len(minima):
            # repeat the minima
            samples = torch.cat([minima] * ((size // len(minima)) + 1), dim=0)
        # add random noise to the samples
        samples = samples + torch.randn_like(samples) * 0.01
        samples = samples[:size]
        return samples

    def setup_test_set(self):
        """Returns a test set of 2D points (dihedral angles).
        Is a 2d grid of points in the range [-pi, pi] x [-pi, pi].
        """
        # return self._dataset_from_minima(self.test_set_size)
        boltzmann_samples, energies = self.sample_boltzmann_distribution(
            num_samples=self.test_set_size,
            temperature=0.01,
        )
        return boltzmann_samples

    def setup_train_set(self):
        """Returns a training set of 2D points (dihedral angles).
        Is a random set of 2D points in the range [-pi, pi] x [-pi, pi].
        """
        # return self._dataset_from_minima(self.train_set_size)
        boltzmann_samples, energies = self.sample_boltzmann_distribution(
            num_samples=self.train_set_size,
            temperature=0.01,
        )
        return boltzmann_samples

    def setup_val_set(self):
        """Returns a validation set of 2D points (dihedral angles).
        Is a 2d grid of points in the range [-pi, pi] x [-pi, pi].
        """
        # return self._dataset_from_minima(self.val_set_size)
        boltzmann_samples, energies = self.sample_boltzmann_distribution(
            num_samples=self.val_set_size,
            temperature=0.01,
        )
        return boltzmann_samples

    def assess_samples(self, samples):
        """Assesses the quality of generated samples.

        Args:
            samples (torch.Tensor): Generated samples
        """
        energies = torch.vmap(self._energy, chunk_size=self.plotting_batch_size)(
            samples
        )
        energy = torch.mean(energies)
        return {"energy": energy}

    # def log_on_epoch_end(
    #     self,
    #     latest_samples: torch.Tensor,
    #     latest_energies: torch.Tensor,
    #     wandb_logger: WandbLogger,
    #     unprioritized_buffer_samples=None,
    #     cfm_samples=None,
    #     replay_buffer=None,
    #     prefix: str = "",
    # ) -> None:
    #     if wandb_logger is None:
    #         return

    #     if len(prefix) > 0 and prefix[-1] != "/":
    #         prefix += "/"

    #     if self.curr_epoch % self.plot_samples_epoch_period == 0:
    #         if latest_samples is not None:
    #             samples_fig = self.get_dataset_fig(latest_samples)
    #             wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

    #     self.curr_epoch += 1

    # def get_dataset_fig(self, samples):
    #     fig, ax = plt.subplots(figsize=(10, 6))

    #     # Plot the normalized energy function
    #     x = torch.linspace(-np.pi, np.pi, 1000, device=self.device)  # phi
    #     y = torch.linspace(-np.pi, np.pi, 1000, device=self.device)  # psi
    #     x, y = torch.meshgrid(x, y, indexing="ij")
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     z = self(torch.cat([x, y], dim=-1))
    #     z = z.reshape(x.shape[0], -1)

    #     # Normalize using trapezoidal rule
    #     dx = x[1] - x[0]
    #     Z = torch.trapz(z, x)
    #     z = z / Z

    #     ax.plot(x.cpu(), z.cpu(), "b-", label="Target Distribution")

    #     # Plot the histogram of samples
    #     if samples is not None:
    #         samples = samples.squeeze(-1)
    #         ax.hist(
    #             samples.cpu(),
    #             bins=50,
    #             density=True,
    #             alpha=0.5,
    #             color="r",
    #             label="Samples",
    #         )

    #     ax.set_xlabel("phi")
    #     ax.set_ylabel("psi")
    #     ax.legend()
    #     ax.grid(True)

    #     return fig_to_image(fig)

    def find_energy_minima(
        self, num_samples=1000, num_minima=5, optimization_steps=100, learning_rate=0.01
    ):
        """
        Find the energy minima of the alanine dipeptide energy landscape.

        Args:
            num_samples (int): Number of random initial points to sample
            num_minima (int): Maximum number of distinct minima to return
            optimization_steps (int): Number of gradient descent steps
            learning_rate (float): Learning rate for gradient descent

        Returns:
            torch.Tensor: Tensor of shape (num_minima, 2) containing the phi/psi coordinates of the minima
        """
        # Generate random initial points in the phi/psi space
        initial_points = (
            torch.rand(num_samples, 2, device=self.device) * 2 * np.pi - np.pi
        )
        initial_points.requires_grad_(True)

        # Perform gradient descent to find minima
        minima = []
        for point in tqdm(initial_points, desc="Finding minima", total=num_samples):
            point_copy = point.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([point_copy], lr=learning_rate)

            for _ in range(optimization_steps):
                optimizer.zero_grad()
                # self returns log(p(x))=-energy, so we negate it to find minima
                energy = -self(point_copy.unsqueeze(0))
                energy.backward()
                optimizer.step()

                # Wrap angles to [-pi, pi]
                with torch.no_grad():
                    point_copy.data = (
                        torch.remainder(point_copy.data + np.pi, 2 * np.pi) - np.pi
                    )

            # Add to minima list if it's a new minimum
            min_point = point_copy.detach()
            tqdm.write(f"Found minimum at {min_point}")
            energy_value = -self(min_point.unsqueeze(0)).item()

            # Check if this is a new minimum (not close to existing ones)
            is_new_minimum = True
            for existing_min in minima:
                existing_point, _ = existing_min
                distance = torch.norm(min_point - existing_point)
                if (
                    distance < 0.1
                ):  # Threshold for considering points as the same minimum
                    is_new_minimum = False
                    break

            if is_new_minimum:
                minima.append((min_point, energy_value))

        # Sort by energy value and take the top num_minima
        minima.sort(key=lambda x: x[1])
        minima = minima[:num_minima]

        # Extract just the coordinates
        minima_coords = torch.stack([m[0] for m in minima])

        print(f"Found {len(minima_coords)} minima")
        print(f"Minima coordinates: \n{minima_coords}")

        return minima_coords

    def get_minima(self, grid_width=200, num_minima=5, device=None):
        """
        Get energy minima by finding the lowest energy points on a grid.

        Args:
            bounds (tuple): Bounds for the grid search as (min, max)
            grid_width (int): Number of points in each dimension for grid
            num_minima (int): Maximum number of minima to return
            device (torch.device): Device to use for computation

        Returns:
            torch.Tensor: Tensor of shape (num_minima, 2) containing the phi/psi coordinates of the minima
        """
        # bounds = self._plotting_bounds
        # load_path = self.get_load_path(bounds=bounds, grid_width=grid_width)
        # with open(f"{load_path}.pkl", "rb") as f:
        #     saved_data = pickle.load(f)

        # log_p_x = saved_data["log_p_x"]
        # energies = -log_p_x

        # x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width, device=device)
        # x_points_dim2 = x_points_dim1
        # x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)), device=device)

        # # Find the indices of the lowest energy points
        # flat_indices = torch.argsort(energies)[:num_minima]

        # # Get the corresponding coordinates
        # minima_coords = x_points[flat_indices]

        # # Get the energy values for these minima
        # minima_energies = energies[flat_indices]

        # print(f"Found {len(minima_coords)} minima")
        # print(f"Minima coordinates: \n{minima_coords}")
        # print(f"Minima energies: \n{minima_energies}")

        # return minima_coords

        # Try to load from cache file if provided
        cache_file = f"dem_outputs/{self.name}/alanine_dipeptide_2d_minima.pt"
        try:
            minima_coords = torch.load(cache_file, weights_only=True)
            # print(f"Info: Loaded {len(minima_coords)} minima from {cache_file}")
            return minima_coords
        except (FileNotFoundError, IOError):
            print(
                f"Info: Cache file {cache_file} not found or invalid. Computing minima..."
            )
        # Compute minima
        minima_coords = self.find_energy_minima()
        # Save to cache file if provided
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(minima_coords, cache_file)
        # print(f"Saved {len(minima_coords)} minima to {cache_file}")
        return minima_coords
