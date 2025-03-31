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

from dem.energies.alanine_dipeptide_dihedral_energy import VectorizedMACE


# TODO: look at lennard jones energy function for all atoms
class MaceAlDiEnergy2D(BaseEnergyFunction):
    """
    Energy function for the alanine dipeptide.
    """

    def __init__(
        self,
        dimensionality=22 * 3,
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
        #
        *args,
        **kwargs,
    ):
        torch.manual_seed(0)
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

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
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Takes in a samples of phi/psi values and returns the energies."""
        minibatch = self.atoms_calc._clone_batch(self.singlebatch_base)
        # minibatch = singlebatch_base.clone()
        # positions_list = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0) # [B, N_atoms, 3]

        def _dihedrals_to_energies(
            positions, node_attrs, edge_index, batch, head, shifts, ptr
        ):
            # samples = positions [N, 3]

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
        # positions = minibatch["positions"].to(samples.device)
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
            in_dims=(0, None, None, None, None, None, None),
        )
        energies = _dihedrals_to_energies_vmapped(
            samples, node_attrs, edge_index, batch, head, shifts, ptr
        )
        energies = energies.squeeze(-1)

        if return_aux_output:
            aux_output = {}
            return energies, aux_output
        return energies

    def _energy_batched(
        self, samples: torch.Tensor, return_aux_output: bool = False
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

        if return_aux_output:
            aux_output = {}
            return energy, aux_output
        return energy

    def _energy(
        self,
        samples: torch.Tensor,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        """Evaluates the pseudoenergy function at given samples.

        Args:
            samples (torch.Tensor): Input points (dihedral angles)

        Returns:
            torch.Tensor: Energy values at input points
        """
        if self.use_vmap:
            return self._energy_vmap(samples, return_aux_output=return_aux_output)
        else:
            return self._energy_batched(samples, return_aux_output=return_aux_output)

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
            log_prob, aux_output = self._energy(samples, return_aux_output=True)
            return log_prob / temperature, aux_output
        return -1 * self._energy(samples) / temperature

    #####################################################################################

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

    def sample_test_set(self, batch_size: int, full: bool = False) -> torch.Tensor:
        """
        Sample a batch of test set data.
        """
        # return super().sample_test_set(batch_size, full)
        return None

    def sample_val_set(self, batch_size: int, full: bool = False) -> torch.Tensor:
        """
        Sample a batch of validation set data.
        """
        # return super().sample_val_set(batch_size, full)
        return None

    def sample_train_set(self, batch_size: int, full: bool = False) -> torch.Tensor:
        return None

    def setup_train_set(self):
        return None

    def setup_val_set(self):
        return None

    def setup_test_set(self):
        return None
