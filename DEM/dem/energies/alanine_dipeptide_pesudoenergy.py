from typing import Optional

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

from alanine_dipeptide.mace_neighbourhood import update_neighborhood_graph_batched
from alanine_dipeptide.alanine_dipeptide_mace import (
    load_alanine_dipeptide_ase,
    repeated_atoms_to_batch,
    update_alanine_dipeptide_xyz_from_dihedrals_batched,
)


class MaceAlDiForcePseudoEnergy(BaseEnergyFunction):
    """
    Pesudoenergy function for the aldanine dipeptide.
    Pseudonergy is defined as the absolute value of the force on the alanine dipeptide.
    Energy and forces are computed using MACE.
    """

    def __init__(
        self,
        dimensionality=2,
        device="cuda",
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=2.0,  # ?
        # loss weights
        force_weight=1.0,
        energy_weight=0.1,
        # Mace
        use_cueq=True,
        dtypestr="float32",
        use_vmap=True,
        batch_size=1,
        #
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
    ):
        torch.manual_seed(0)
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.curr_epoch = 0
        self.name = "aldanine_dipeptide_mace_pesudoenergy"

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.batch_size = batch_size
        self.use_vmap = use_vmap

        ######################################################
        # Get MACE model
        ######################################################
        # mace_off or mace_anicc
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(
            model="medium", device=device_str, dtype=dtypestr, enable_cueq=use_cueq
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
        self.batch_base = batch_base.to_dict()
        self.model = atoms_calc.models[0]
        self.atoms_calc = atoms_calc
        self.atoms = atoms

        # Make minibatch version of batch, that is just multiple copies of the same AlDi configuration
        # but mimicks a batch from a typical torch_geometric dataloader
        # and that we can then rotate to get different phi/psi values
        if self.use_vmap:
            # batch_size of one that is duplicated by vmap
            self.minibatch = repeated_atoms_to_batch(
                self.atoms_calc, copy.deepcopy(self.atoms), bs=1, repeats=1
            )
        else:
            self.minibatch = repeated_atoms_to_batch(
                self.atoms_calc,
                copy.deepcopy(self.atoms),
                bs=batch_size,
                repeats=batch_size,
            )

        ######################################################
        # Initialize BaseEnergyFunction
        ######################################################
        super().__init__(
            dimensionality=2,
            is_molecule=True,
            # normalization for the force?
            normalization_min=-4000,
            normalization_max=-3000,  # TODO: change this
        )

    def setup_test_set(self):
        """Returns a test set of 2D points (dihedral angles).
        Is a 2d grid of points in the range [-pi, pi] x [-pi, pi].
        """
        samples = torch.linspace(-np.pi, np.pi, self.test_set_size, device=self.device)
        samples = torch.cartesian_prod(samples, samples)
        return samples

    def setup_train_set(self):
        """Returns a training set of 2D points (dihedral angles).
        Is a random set of 2D points in the range [-pi, pi] x [-pi, pi].
        """
        samples = (
            torch.rand(self.train_set_size, 2, device=self.device) * 2 * np.pi - np.pi
        )
        # samples = torch.cartesian_prod(samples, samples)
        return samples

    def setup_val_set(self):
        """Returns a validation set of 2D points (dihedral angles).
        Is a 2d grid of points in the range [-pi, pi] x [-pi, pi].
        """
        samples = torch.linspace(-np.pi, np.pi, self.val_set_size, device=self.device)
        samples = torch.cartesian_prod(samples, samples)
        return samples

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Evaluates the pseudoenergy function at given samples.

        Args:
            samples (torch.Tensor): Input points (dihedral angles)

        Returns:
            torch.Tensor: Energy values at input points
        """
        if self.use_vmap:
            return self._forward_vmap(samples)
        else:
            return self._forward_batched(samples)

    def _forward_vmap(self, samples: torch.Tensor) -> torch.Tensor:
        """Takes in a single sample and returns the pseudoenergy.
        No batch dimension
        """
        minibatch = self.minibatch.clone()

        # TODO: does this do anything?
        if self.atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            minibatch = self.atoms_calc._clone_batch(minibatch)
            node_heads = minibatch["head"][minibatch["batch"]]
            num_atoms_arange = torch.arange(minibatch["positions"].shape[0])
            node_e0 = self.model.atomic_energies_fn(minibatch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.atoms_calc.use_compile
        else:
            compute_stress = False

        # Update xyz positions of atoms based on phi/psi values
        # forces = gradient of energy with respect to phi/psi
        phi_psi_batch = samples.requires_grad_(True)
        # TODO: code up non-batched version
        minibatch = update_alanine_dipeptide_xyz_from_dihedrals_batched(
            phi_psi_batch, minibatch, convention=convention
        )

        # Update edge indices
        # TODO: why is the cell so huge?
        minibatch = update_neighborhood_graph_batched(
            minibatch, model.r_max.item(), overwrite_cell=True
        )

        # TODO: non-batched version
        # Compute energies and forces
        out = self.model(minibatch, compute_stress=compute_stress, training=True)
        # [B, 2]
        forces = torch.autograd.grad(
            outputs=out["energy"],  # [B]
            inputs=phi_psi_batch,  # [B, 2]
            grad_outputs=torch.ones_like(out["energy"]),  # [B]
            create_graph=True,
        )[0]

        forces_norm = torch.linalg.norm(forces, dim=1)

        # TODO: normalize energies and maybe forces to a reasonable range
        energy_loss = (self.energy_weight * out["energy"]).mean()
        force_loss = (self.force_weight * forces_norm).mean()
        total_loss = energy_loss + force_loss
        if return_aux_output:
            aux_output = {
                "energy_loss": energy_loss,
                "force_loss": force_loss,
                "total_loss": total_loss,
            }
            return total_loss, aux_output
        return total_loss

    def _forward_batched(self, samples: torch.Tensor) -> torch.Tensor:
        bs = samples.shape[0]
        if bs == self.batch_size:
            minibatch = self.minibatch.clone()
        else:
            # construct a minibatch with the correct batch size
            self.minibatch = repeated_atoms_to_batch(
                self.atoms_calc, copy.deepcopy(self.atoms), bs=bs, repeats=bs
            )

        # TODO: does this do anything?
        if self.atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            minibatch = self.atoms_calc._clone_batch(minibatch)
            node_heads = minibatch["head"][minibatch["batch"]]
            num_atoms_arange = torch.arange(minibatch["positions"].shape[0])
            node_e0 = self.model.atomic_energies_fn(minibatch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.atoms_calc.use_compile
        else:
            compute_stress = False

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

        # Compute energies and forces for all configurations
        out = self.model(minibatch, compute_stress=compute_stress, training=True)
        # [B, 2]
        forces = torch.autograd.grad(
            outputs=out["energy"],  # [B]
            inputs=phi_psi_batch,  # [B, 2]
            grad_outputs=torch.ones_like(out["energy"]),  # [B]
            create_graph=True,
        )[0]

        forces_norm = torch.linalg.norm(forces, dim=1)

        # TODO: normalize energies and maybe forces to a reasonable range
        pseudoenergy = (
            self.energy_weight * out["energy"] + self.force_weight * forces_norm
        )
        return pseudoenergy

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if latest_samples is not None:
                samples_fig = self.get_dataset_fig(latest_samples)
                wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

        self.curr_epoch += 1

    def get_dataset_fig(self, samples):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the normalized energy function
        x = torch.linspace(-np.pi, np.pi, 1000, device=self.device)  # phi
        y = torch.linspace(-np.pi, np.pi, 1000, device=self.device)  # psi
        x, y = torch.meshgrid(x, y, indexing="ij")
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = self(torch.cat([x, y], dim=-1))
        z = z.reshape(x.shape[0], -1)

        # Normalize using trapezoidal rule
        dx = x[1] - x[0]
        Z = torch.trapz(z, x)
        z = z / Z

        ax.plot(x.cpu(), z.cpu(), "b-", label="Target Distribution")

        # Plot the histogram of samples
        if samples is not None:
            samples = samples.squeeze(-1)
            ax.hist(
                samples.cpu(),
                bins=50,
                density=True,
                alpha=0.5,
                color="r",
                label="Samples",
            )

        ax.set_xlabel("phi")
        ax.set_ylabel("psi")
        ax.legend()
        ax.grid(True)

        return fig_to_image(fig)
