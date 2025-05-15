from typing import Union, Optional, Sequence
from collections.abc import Sequence as _Sequence

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
# from bgflow import MultiDoubleWellPotential
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger
import os

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.data_utils import remove_mean
from dem.utils.geometry import compute_distances

def _is_non_empty_sequence_of_integers(x):
    return (
        isinstance(x, _Sequence) and (len(x) > 0) and all(isinstance(y, int) for y in x)
    )


def _is_sequence_of_non_empty_sequences_of_integers(x):
    return (
        isinstance(x, _Sequence)
        and len(x) > 0
        and all(_is_non_empty_sequence_of_integers(y) for y in x)
    )


def _parse_dim(dim):
    if isinstance(dim, int):
        return [torch.Size([dim])]
    if _is_non_empty_sequence_of_integers(dim):
        return [torch.Size(dim)]
    elif _is_sequence_of_non_empty_sequences_of_integers(dim):
        return list(map(torch.Size, dim))
    else:
        raise ValueError(
            f"dim must be either:"
            f"\n\t- an integer"
            f"\n\t- a non-empty list of integers"
            f"\n\t- a list with len > 1 containing non-empty lists containing integers"
        )


# https://github.com/noegroup/bgflow/blob/main/bgflow/distribution/energy/base.py
class MultiDoubleWellPotential(torch.nn.Module):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        super().__init__(**kwargs)
        if two_event_dims:
            self._event_shapes = _parse_dim([n_particles, dim // n_particles])
        else:
            self._event_shapes = _parse_dim(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)

    def energy(self, *xs, temperature=1.0, **kwargs):
        assert len(xs) == len(
            self._event_shapes
        ), f"Expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        batch_shape = xs[0].shape[: -len(self._event_shapes[0])]
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert x.shape[: -len(s)] == batch_shape, (
                f"Inconsistent batch shapes."
                f"Input at index {i} has batch shape {x.shape[:-len(s)]}"
                f"however input at index 0 has batch shape {batch_shape}."
            )
            assert (
                x.shape[-len(s) :] == s
            ), f"Input at index {i} as wrong shape {x.shape[-len(s):]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shapes[0]) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return int(torch.prod(torch.tensor(self.event_shape, dtype=int)))

    @property
    def event_shape(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore therefore there exists no single event shape."
                "Consider using Energy.event_shapes instead."
            )
        return self._event_shapes[0]

    @property
    def event_shapes(self):
        return self._event_shapes


class MultiDoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        data_from_efm=True,  # if False, data from EFM
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=True,
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.data_from_efm = data_from_efm

        if data_from_efm:
            self.name = "DW4_EFM"
        else:
            self.name = "DW4_EACF"

        if self.data_from_efm:
            if data_path_train is None:
                raise ValueError("DW4 is from EFM. No train data path provided")
            if data_path_val is None:
                raise ValueError("DW4 is from EFM. No val data path provided")

        # self.data_path = get_original_cwd() + "/" + data_path
        # self.data_path_train = get_original_cwd() + "/" + data_path_train
        # self.data_path_val = get_original_cwd() + "/" + data_path_val

        self.data_path = data_path
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val

        self.device = device

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 0

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dimensionality,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        return self.log_prob(samples, return_aux_output)

    def log_prob(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        if return_aux_output:
            aux_output = {}
            return -self.multi_double_well.energy(samples).squeeze(-1), aux_output
        return -self.multi_double_well.energy(samples).squeeze(-1)

    def setup_test_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][-self.test_set_size :]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_train_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_train, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][: self.train_set_size]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )

        return data

    def setup_val_set(self):
        if self.data_from_efm:
            data = np.load(self.data_path_val, allow_pickle=True)

        else:
            all_data = np.load(self.data_path, allow_pickle=True)
            data = all_data[0][
                -self.test_set_size - self.val_set_size : -self.test_set_size
            ]
            del all_data

        data = remove_mean(torch.tensor(data), self.n_particles, self.n_spatial_dim).to(
            self.device
        )
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1)
            == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples = self.unnormalize(samples)
        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

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
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(
                    f"{prefix}cfm_generated_samples", [cfm_samples_fig]
                )

        self.curr_epoch += 1

    def get_dataset_fig(self, samples, random_samples=False):
        test_data_smaller = self.sample_test_set(1000)

        if random_samples:
            # normal distribution
            samples = torch.randn(test_data_smaller.shape).to(test_data_smaller.device)
            # samples = test_data_smaller

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        if samples is not None:
            dist_samples = self.interatomic_dist(samples).detach().cpu()
            axs[0].hist(
                dist_samples.view(-1),
                bins=100,
                alpha=0.5,
                density=True,
                histtype="step",
                linewidth=4,
            )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["test data", "generated data"])

        energy_test = -self(test_data_smaller).detach().detach().cpu()

        min_energy = -26
        max_energy = 0

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        if samples is not None:
            energy_samples = -self(samples).detach().detach().cpu()
            axs[1].hist(
                energy_samples.cpu(),
                bins=100,
                density=True,
                alpha=0.4,
                range=(min_energy, max_energy),
                color="r",
                histtype="step",
                linewidth=4,
                label="generated data",
            )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )

        # try:
        #     buffer = BytesIO()
        #     fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        #     buffer.seek(0)

        #     return PIL.Image.open(buffer)

        # except Exception as e:
        #     fig.canvas.draw()
        #     return PIL.Image.frombytes(
        #         "RGB", fig.canvas.get_width_height(), fig.canvas.renderer.buffer_rgba()
        #     )
        #     fig.canvas.draw()
        #     return PIL.Image.frombytes(
        #         "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        #     )
