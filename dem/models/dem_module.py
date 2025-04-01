import math
import random
import time
from typing import Any, Dict, Optional

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric
from tqdm import tqdm
import os
import traceback

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import (
    compute_distribution_distances,
    compute_full_dataset_distribution_distances,
)
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.mlp import TimeConder
from .components.noise_schedules import BaseNoiseSchedule
from .components.temperature_schedule import BaseTemperatureSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt, wrap_for_richardsons
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde, integrate_constrained_sde
from .components.sdes import VEReverseSDE


class ProjectedVectorField:
    """Wrapper class to project out components of a vector field."""

    def __init__(self, base_model, projection_mask=None):
        """
        Args:
            base_model: Base model that computes the vector field
            projection_mask: Boolean mask of shape matching output dimension.
                           True indicates components to keep, False to zero out.
        """
        self.base_model = base_model
        self.projection_mask = projection_mask

    def __call__(self, t, x):
        vector_field = self.base_model(t, x)
        if self.projection_mask is not None:
            # Expand mask to match batch dimension
            expanded_mask = self.projection_mask.expand(vector_field.shape[0], -1)
            vector_field = vector_field * expanded_mask
        return vector_field


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning time values.

    Args:
        batch_t: Batch of time values
        batch_loss: Batch of loss values
        num_bins: Number of bins to stratify into
        loss_name: Name prefix for the loss

    Returns:
        Dictionary mapping time ranges to average loss in that range
    """
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start: .2f}, {bin_end: .2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the Weights & Biases logger from a list of loggers.

    Args:
        loggers: List of PyTorch Lightning loggers

    Returns:
        The WandbLogger instance if present, otherwise None
    """
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class DEMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        temperature_schedule: BaseTemperatureSchedule,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_buffer_samples_to_generate_init: int,
        num_estimator_mc_samples: int,
        num_buffer_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        use_richardson_in_trainloss: bool = False,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_buffer_from_prior=False,
        init_buffer_from_train=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
        seed=None,
        nll_batch_size=256,
        # added
        use_vmap=True,
        gen_batch_size: int = -1,
        num_samples_to_plot: int = 256,
        generate_constrained_samples: bool = False,
        constrained_score_norm_target: float = 0.0,
        force_grad: bool = False,
        buffer_temperature: str = "same",
        val_temperature: str = "same",
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)

        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:
            self.net = ScalingWrapper(
                self.net, input_scaling_factor, output_scaling_factor
            )

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )

        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)

            self.net = self.score_scaler.wrap_model_for_unscaling(self.net)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)

        # TODO: pass temperature schedule?
        self.dem_cnf = CNF(
            self.net,
            is_diffusion=True,
            use_exact_likelihood=use_exact_likelihood,
            noise_schedule=noise_schedule,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )
        self.cfm_cnf = CNF(
            self.cfm_net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )

        self.nll_with_cfm = nll_with_cfm
        self.nll_with_dem = nll_with_dem
        self.nll_on_buffer = nll_on_buffer
        self.logz_with_cfm = logz_with_cfm
        self.cfm_prior_std = cfm_prior_std
        self.compute_nll_on_train_data = compute_nll_on_train_data
        self.nll_batch_size = nll_batch_size

        flow_matcher = ConditionalFlowMatcher
        if use_otcfm:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher

        self.cfm_sigma = cfm_sigma
        self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

        self.nll_integration_method = nll_integration_method

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.temperature_schedule = temperature_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality

        self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule)

        grad_fxn = estimate_grad_Rt
        if use_richardsons:
            grad_fxn = wrap_for_richardsons(grad_fxn)
        
        # TODO: added by Andreas because DEM just used estimate_grad_Rt in trainloss?
        self.use_richardson_in_trainloss = use_richardson_in_trainloss
        if self.use_richardson_in_trainloss:
            self.grad_fxn_trainloss = grad_fxn
        else:
            self.grad_fxn_trainloss = estimate_grad_Rt

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)

        self.dem_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_nll_logdetjac = MeanMetric()
        self.test_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.test_nfe = MeanMetric()
        self.val_energy_w2 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()

        self.val_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()
        self.val_logz = MeanMetric()
        self.val_big_batch_logz = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()
        self.val_dem_ess = MeanMetric()
        self.val_ess = MeanMetric()
        self.val_big_batch_ess = MeanMetric()
        self.test_dem_ess = MeanMetric()
        self.test_ess = MeanMetric()

        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()
        self.val_buffer_ess = MeanMetric()
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()
        self.test_buffer_ess = MeanMetric()

        self.val_train_nll_logdetjac = MeanMetric()
        self.val_train_nll_log_p_1 = MeanMetric()
        self.val_train_nll = MeanMetric()
        self.val_train_nfe = MeanMetric()
        self.val_train_logz = MeanMetric()
        self.val_train_ess = MeanMetric()
        self.test_train_nll_logdetjac = MeanMetric()
        self.test_train_nll_log_p_1 = MeanMetric()
        self.test_train_nll = MeanMetric()
        self.test_train_nfe = MeanMetric()
        self.test_train_logz = MeanMetric()
        self.test_train_ess = MeanMetric()

        self.num_buffer_samples_to_generate_init = num_buffer_samples_to_generate_init
        # number of samples to de-/noise in loss. corresponds to batch size
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        # number of MC samples per batch for score estimation in loss
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_buffer_samples_to_generate_per_epoch = num_buffer_samples_to_generate_per_epoch
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        self.eval_batch_size = eval_batch_size
        self.gen_batch_size = gen_batch_size
        self.num_samples_to_plot = num_samples_to_plot
        
        self.prioritize_cfm_training_samples = prioritize_cfm_training_samples
        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.diffusion_scale = diffusion_scale
        self.init_buffer_from_prior = init_buffer_from_prior
        self.init_buffer_from_train = init_buffer_from_train

        self.negative_time = negative_time
        self.num_negative_time_steps = num_negative_time_steps
        self.use_vmap = use_vmap
        assert (
            isinstance(buffer_temperature, float) or buffer_temperature == "same"
        ), f"buffer_temperature must be a float or 'same', got {buffer_temperature}"
        self.buffer_temperature = buffer_temperature
        assert (
            isinstance(val_temperature, float) or val_temperature == "same"
        ), f"val_temperature must be a float or 'same', got {val_temperature}"
        self.val_temperature = val_temperature

        self.generate_constrained_samples = generate_constrained_samples
        self.constrained_score_norm_target = constrained_score_norm_target
        self.force_grad = force_grad

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        x1 = samples
        x1 = self.energy_function.unnormalize(x1)

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(
            x0, x1
        )

        if self.energy_function.is_molecule and self.cfm_sigma != 0:
            xt = remove_mean(
                xt, self.energy_function.n_particles, self.energy_function.n_spatial_dim
            )

        vt = self.cfm_net(t, xt)
        loss = (vt - ut).pow(2).mean(dim=-1)

        # if self.energy_function.normalization_max is not None:
        #    loss = loss / (self.energy_function.normalization_max ** 2)

        return loss

    def should_train_cfm(self, batch_idx: int) -> bool:
        """Use Conditional Flow Matching (CFM) to train the model."""
        return self.nll_with_cfm or self.hparams.debug_use_train_data

    def get_score_loss(
        self, times: torch.Tensor, samples: torch.Tensor, noised_samples: torch.Tensor
    ) -> torch.Tensor:
        predicted_score = self.forward(times, noised_samples)

        true_score = -(noised_samples - samples) / (
            self.noise_schedule.h(times).unsqueeze(1) + 1e-4
        )
        error_norms = (predicted_score - true_score).pow(2).mean(-1)
        return error_norms

    def get_loss(
        self,
        times: torch.Tensor,
        samples: torch.Tensor,
        return_aux_output: bool = False,
    ) -> torch.Tensor:
        # B = num_samples_to_sample_from_buffer
        # samples [B, D] times [B]

        # By default did not use Richardson extrapolation (estimate_grad_Rt) 
        # instead of grad_fxn
        _out = self.grad_fxn_trainloss(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_estimator_mc_samples,
            # use_vmap=self.use_vmap,
            return_aux_output=return_aux_output,
            temperature=self.temperature_schedule(self.global_step),
        )
        # estimated_score: [B, D]
        if return_aux_output:
            estimated_score, aux_output = _out
        else:
            estimated_score = _out
            
        # score_norm: [B]
        score_norm = (estimated_score.detach() ** 2).mean(-1).sqrt()

        if self.clipper is not None and self.clipper.should_clip_scores:
            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            estimated_score = self.clipper.clip_scores(estimated_score)

            if self.energy_function.is_molecule:
                estimated_score = estimated_score.reshape(
                    -1, self.energy_function.dimensionality
                )

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(
                estimated_score, times
            )

        predicted_score = self.forward(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)

        if return_aux_output:
            aux_output["avg_score_norm"] = score_norm.mean()
            aux_output["max_score_norm"] = score_norm.max()
            aux_output["min_score_norm"] = score_norm.min()
            return self.lambda_weighter(times) * error_norms, aux_output
        else:
            return self.lambda_weighter(times) * error_norms

    def training_step(self, batch, batch_idx):
        """Samples from the buffer (iDEM) or prior (pDEM), noises and denoises, computes the loss."""
        loss = 0.0
        if not self.hparams.debug_use_train_data:
            if self.hparams.use_buffer:
                # iDEM with buffer
                iter_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer
                )
            else:
                # prior DEM (pDEM) without buffer, truly simulation free
                iter_samples = self.prior.sample(self.num_samples_to_sample_from_buffer)
                # Uncomment for score matching (SM)
                # iter_samples = self.energy_function.sample_train_set(self.num_samples_to_sample_from_buffer)

            times = torch.rand(
                (self.num_samples_to_sample_from_buffer,), device=iter_samples.device
            )

            noised_samples = iter_samples + (
                torch.randn_like(iter_samples)
                * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
            )

            if self.energy_function.is_molecule:
                noised_samples = remove_mean(
                    noised_samples,
                    self.energy_function.n_particles,
                    self.energy_function.n_spatial_dim,
                )

            dem_loss, aux_output = self.get_loss(
                times, noised_samples, return_aux_output=True
            )
            # assert torch.isfinite(
            #     dem_loss
            # ).all(), f"{(~torch.isfinite(dem_loss)).sum().item()} entries are NaN/inf. Epoch={self.current_epoch}, step={self.global_step}"
            # for k, v in aux_output.items():
            #     assert torch.isfinite(
            #         v
            #     ).all(), f"NaN/inf in {k}\n{v}. Epoch={self.current_epoch}, step={self.global_step}"

            # Uncomment for SM
            # dem_loss = self.get_score_loss(times, iter_samples, noised_samples)
            self.log_dict(
                t_stratified_loss(
                    times, dem_loss, loss_name="train/stratified/dem_loss"
                )
            )
            dem_loss = dem_loss.mean()
            loss = loss + dem_loss

            # update and log metrics
            self.dem_train_loss(dem_loss)
            self.log(
                "train/dem_loss",
                self.dem_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            for key, value in aux_output.items():
                self.log(
                    f"train/dem_{key}",
                    value.mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

        if self.should_train_cfm(batch_idx):
            if self.hparams.debug_use_train_data:
                # use conditional flow matching to train on train set
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,), device=cfm_samples.device
                )
            else:
                # use buffer samples for training
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )

            cfm_loss = self.get_cfm_loss(cfm_samples)
            
            self.log_dict(
                t_stratified_loss(
                    times, cfm_loss, loss_name="train/stratified/cfm_loss"
                )
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
        
        try:
            optimizer = self.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            lr = optimizer.param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False)
        except Exception as e:
            print(f"Error logging learning rate: {e}")
            pass

        try:
            temperature = self.temperature_schedule(self.global_step)
            self.log("train/temperature", temperature, on_step=True, on_epoch=False, prog_bar=False)
        except Exception as e:
            print(f"Error logging temperature: {e}")
            pass
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
        negative_time=False,
        projection_mask=None,
        temperature=1.0,
        batch_size=-1,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_buffer_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)

        return self.integrate(
            reverse_sde=reverse_sde,
            samples=samples,
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            negative_time=negative_time,
            projection_mask=projection_mask,
            temperature=temperature,
            batch_size=batch_size,
        )

    def integrate(
        self,
        reverse_sde: VEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        negative_time=False,
        projection_mask=None,
        constrain_score_norm=False,
        constrained_score_norm_target=0.0,
        temperature=1.0,
        batch_size=-1,
    ) -> torch.Tensor:

        if reverse_sde is None:
            if projection_mask is not None:
                # Wrap the model's vector field with projection
                projected_model = ProjectedVectorField(self.net, projection_mask)
                reverse_sde = VEReverseSDE(projected_model, self.noise_schedule)
            else:
                reverse_sde = self.reverse_sde

        # Kirill WIP
        # if constrain_score_norm:
        #     trajectory = integrate_constrained_sde(
        #         sde=reverse_sde,
        #         x0=samples,
        #         num_integration_steps=self.num_integration_steps,
        #         energy_function=self.energy_function,
        #         constant_of_motion_fn=None,  # TODO: add constant of motion
        #         diffusion_scale=diffusion_scale,
        #         reverse_time=reverse_time,
        #         no_grad=no_grad,
        #         negative_time=negative_time,
        #         num_negative_time_steps=self.num_negative_time_steps,
        #         clipper=self.clipper,
        #     )
        # else:

        # added for pseudopotentials that relies on gradients
        if self.force_grad:
            no_grad = False
            
        # samples: [B, D] or [B, N, D]
        # trajectory: [T, B, D] 
        trajectory = integrate_sde(
            sde=reverse_sde,
            x0=samples,
            num_integration_steps=self.num_integration_steps,
            energy_function=self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            negative_time=negative_time,
            num_negative_time_steps=self.num_negative_time_steps,
            clipper=self.clipper,
            temperature=temperature,
            batch_size=batch_size,
        )
            
        if return_full_trajectory:
            return trajectory

        # [B, D] 
        return trajectory[-1]

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        batch_size = self.nll_batch_size
        num_batches = math.ceil(len(samples) / float(batch_size))
        nlls, x_1s, logdetjacs, log_p_1s = [], [], [], []
        for i in tqdm(range(num_batches), desc="Computing NLL"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            iter_samples = samples[start_idx:end_idx]

            aug_samples = torch.cat(
                [
                    iter_samples,
                    torch.zeros(iter_samples.shape[0], 1, device=samples.device),
                ],
                dim=-1,
            )
            aug_output = cnf.integrate(aug_samples)[-1]
            x_1s.append(aug_output[..., :-1])
            logdetjacs.append(aug_output[..., -1])

            log_p_1s.append(prior.log_prob(x_1s[-1]))
            log_p_0 = log_p_1s[-1] + logdetjacs[-1]

            nlls.append(-log_p_0)

        return (
            torch.cat(nlls, dim=0),
            torch.cat(x_1s, dim=0),
            torch.cat(logdetjacs, dim=0),
            torch.cat(log_p_1s, dim=0),
        )

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.buffer_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.buffer_temperature

        if self.clipper_gen is not None:
            reverse_sde = VEReverseSDE(
                self.clipper_gen.wrap_grad_fxn(self.net), self.noise_schedule
            )
            self.last_samples = self.generate_samples(
                reverse_sde=reverse_sde,
                diffusion_scale=self.diffusion_scale,
                temperature=temperature,
                num_samples=self.num_samples_to_plot,
                batch_size=self.gen_batch_size,
            )
            self.last_energies = self.energy_function(self.last_samples)
        else:
            self.last_samples = self.generate_samples(
                diffusion_scale=self.diffusion_scale,
                temperature=temperature,
                num_samples=self.num_samples_to_plot,
                batch_size=self.gen_batch_size,
            )
            self.last_energies = self.energy_function(self.last_samples)

        self.buffer.add(self.last_samples, self.last_energies)

        self._log_energy_w2(prefix="val")

        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="val")
            self._log_dist_total_var(prefix="val")

    def _log_energy_w2(self, prefix="val"):
        """Wasserstein distance (EMD) between energy of data set and energy of generated samples"""
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                batch_size=self.gen_batch_size,
            )
            generated_energies = self.energy_function(generated_samples)
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            _, generated_energies = self.buffer.get_last_n_inserted(
                self.eval_batch_size
            )

        if data_set is None:
            print("Warning: data_set is None skipping energy w2")
            return

        energies = self.energy_function(self.energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(
            energies.cpu().numpy(), generated_energies.cpu().numpy()
        )

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _log_dist_w2(self, prefix="val"):
        if self.val_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.val_temperature

        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                temperature=temperature,
                batch_size=self.gen_batch_size,
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples)
            .cpu()
            .numpy()
            .reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _log_dist_total_var(self, prefix="val"):
        if self.val_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.val_temperature

        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                temperature=temperature,
                batch_size=self.gen_batch_size,
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        total_var = self._compute_total_var(generated_samples, data_set)

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _compute_total_var(self, generated_samples, data_set):
        """Compute total variation distance between histograms of interatomic distances.

        Calculates the total variation distance between normalized histograms of interatomic
        distances from generated samples and a reference dataset.

        Args:
            generated_samples (torch.Tensor): Tensor of generated molecular configurations
            data_set (torch.Tensor): Tensor of reference molecular configurations from training/test set

        Returns:
            float: Total variation distance between the normalized histograms, in range [0,1]
        """
        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples)
            .cpu()
            .numpy()
            .reshape(-1),
        )
        data_set_dists = (
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)
        )

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(
            generated_samples_dists, bins=(x_data_set)
        )
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum()
                - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        return total_var

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, _, _, _ = self.compute_nll(cnf, prior, samples)
        # energy function will unnormalize the samples itself
        logp = self.energy_function(self.energy_function.normalize(samples))
        logq = -nll
        logz = logp - logq

        logz_metric = getattr(self, f"{prefix}_{name}logz")
        logz_metric.update(logz)
        self.log(
            f"{prefix}/{name}logz",
            logz_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        ess = (torch.nn.functional.softmax(logp - logq) ** 2).sum().pow(-1)
        # Normalize to range 0-1
        ess = ess / logp.shape[0]
        ess_metric = getattr(self, f"{prefix}_{name}ess")
        ess_metric.update(ess)
        self.log(
            f"{prefix}/{name}ess",
            ess_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def compute_and_log_nll(self, cnf, prior, samples, prefix, name):
        batch_size = self.nll_batch_size
        num_batches = math.ceil(len(samples) / float(batch_size))

        range_generator = range(num_batches)
        if num_batches > 3:
            range_generator = tqdm(range_generator, desc=f"Computing NLL for {name}")

        for i in range_generator:
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = samples[start_idx:end_idx]

            cnf.nfe = 0.0
            nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(
                cnf, prior, batch
            )
            nfe_metric = getattr(self, f"{prefix}_{name}nfe")
            nll_metric = getattr(self, f"{prefix}_{name}nll")
            logdetjac_metric = getattr(self, f"{prefix}_{name}nll_logdetjac")
            log_p_1_metric = getattr(self, f"{prefix}_{name}nll_log_p_1")
            nfe_metric.update(cnf.nfe)
            nll_metric.update(nll)
            logdetjac_metric.update(logdetjac)
            log_p_1_metric.update(log_p_1)

        self.log_dict(
            {
                f"{prefix}/{name}_nfe": nfe_metric,
                f"{prefix}/{name}nll_logdetjac": logdetjac_metric,
                f"{prefix}/{name}nll_log_p_1": log_p_1_metric,
                # f"{prefix}/{name}logz": logz_metric,
            },
            on_epoch=True,
        )
        self.log(
            f"{prefix}/{name}nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return forwards_samples

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of generated data or from the test/validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # get reference metrics from the test / val set
        if prefix == "test":
            # If we're doing CFM eval none of the actual useful stuff is done in here
            # so just skip it (since NLL eval takes a while)
            if self.nll_with_cfm:
                return 0.0

            batch = self.energy_function.sample_test_set(self.eval_batch_size)
        elif prefix == "val":
            batch = self.energy_function.sample_val_set(self.eval_batch_size)

        backwards_samples = self.last_samples
        if self.val_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.val_temperature

        # generate samples noise --> data if needed
        if backwards_samples is None or self.eval_batch_size > len(backwards_samples):
            backwards_samples = self.generate_samples(
                num_samples=self.eval_batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                temperature=temperature,
                batch_size=self.gen_batch_size,
            )

        # sample eval_batch_size from generated samples from dem to match dimensions
        # required for distribution metrics
        if (
            backwards_samples is not None
            and len(backwards_samples) != self.eval_batch_size
        ):
            indices = torch.randperm(len(backwards_samples))[: self.eval_batch_size]
            backwards_samples = backwards_samples[indices]

        if batch is None:
            print(f"Warning batch is None skipping eval for {prefix}")
            self.eval_step_outputs.append({"gen_0": backwards_samples})
            return

        times = torch.rand((self.eval_batch_size,), device=batch.device)

        noised_batch = batch + (
            torch.randn_like(batch) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
        )

        if self.energy_function.is_molecule:
            noised_batch = remove_mean(
                noised_batch,
                self.energy_function.n_particles,
                self.energy_function.n_spatial_dim,
            )

        loss = self.get_loss(times, noised_batch).mean(-1)

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        self.log(
            f"{prefix}/loss",
            loss_metric,
            on_step=True,
            on_epoch=True,
            # if to print metrics to terminal
            prog_bar=False,
        )

        to_log = {
            "data_0": batch,
            "gen_0": backwards_samples,
        }

        if self.nll_with_dem:
            batch = self.energy_function.normalize(batch)
            forwards_samples = self.compute_and_log_nll(
                self.dem_cnf, self.prior, batch, prefix, "dem_"
            )
            self.compute_log_z(
                self.cfm_cnf, self.prior, backwards_samples, prefix, "dem_"
            )
        if self.nll_with_cfm:
            forwards_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, batch, prefix, ""
            )

            iter_samples, _, _ = self.buffer.sample(self.eval_batch_size)

            # compute nll on buffer if not training cfm only
            if not self.hparams.debug_use_train_data and self.nll_on_buffer:
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, iter_samples, prefix, "buffer_"
                )

            if self.compute_nll_on_train_data:
                train_samples = self.energy_function.sample_train_set(
                    self.eval_batch_size
                )
                forwards_samples = self.compute_and_log_nll(
                    self.cfm_cnf, self.cfm_prior, train_samples, prefix, "train_"
                )

        if self.logz_with_cfm:
            backwards_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(self.eval_batch_size),
            )[-1]
            # backwards_samples = self.generate_cfm_samples(self.eval_batch_size)
            self.compute_log_z(
                self.cfm_cnf, self.cfm_prior, backwards_samples, prefix, ""
            )

        # Add custom validation for convergence to transition states
        if self.energy_function.is_transition_sampler and hasattr(
            self.energy_function, "get_true_transition_states"
        ):
            # Get ground truth transition states
            true_transition_states = self.energy_function.get_true_transition_states()

            # Threshold distance for considering a point "near" a transition state
            distance_threshold = 1.0  # Can be made configurable

            # Compute all pairwise distances
            dists = torch.cdist(backwards_samples, true_transition_states)
            min_dists = dists.min(dim=1)[0]  # Closest transition state to each sample
            min_sample_dists = dists.min(dim=0)[
                0
            ]  # Closest sample to each transition state

            # Coverage Metrics
            transition_states_covered = (
                (min_sample_dists < distance_threshold).float().mean()
            )
            coverage_radius = (
                min_sample_dists.max()
            )  # Minimum radius needed to cover all transition states

            # Precision Metrics
            samples_near_transition = (min_dists < distance_threshold).float().mean()

            # Force magnitude at samples (requires grad)
            with torch.enable_grad():
                samples_grad = backwards_samples.detach().requires_grad_(True)
                energy = -self.energy_function.log_prob(samples_grad)
                forces = torch.autograd.grad(energy.sum(), samples_grad)[0]
                force_magnitudes = forces.norm(dim=-1)
                avg_force = force_magnitudes.mean()

            # Hessian eigenvalue metrics
            def compute_hessian_metrics(x):
                hessian = torch.func.hessian(
                    lambda x: -self.energy_function.log_prob(x)
                )(x)
                eigenvals = torch.linalg.eigvalsh(hessian)
                # Check for index-1 saddle point pattern (one negative, rest positive)
                n_negative = (eigenvals < 0).sum()
                return n_negative == 1

            # Compute Hessian metrics for points near transition states
            close_samples = backwards_samples[min_dists < distance_threshold]
            if len(close_samples) > 0:
                hessian_accuracy = (
                    torch.tensor([compute_hessian_metrics(x) for x in close_samples])
                    .float()
                    .mean()
                )
            else:
                hessian_accuracy = torch.tensor(0.0)

            # Log all metrics
            metrics = {
                # Coverage metrics
                f"{prefix}/transition_states_covered": transition_states_covered,
                f"{prefix}/transition_coverage_radius": coverage_radius,
                # Precision metrics
                f"{prefix}/samples_near_transition": samples_near_transition,
                f"{prefix}/average_force_magnitude": avg_force,
                f"{prefix}/hessian_accuracy": hessian_accuracy,
                # Original distance metrics
                f"{prefix}/transition_state_mean_dist": min_dists.mean(),
                f"{prefix}/transition_state_median_dist": min_dists.median(),
                f"{prefix}/transition_state_min_dist": min_dists.min(),
            }

            # Log all metrics
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)
    
    def get_energies(self, samples, batch_size=-1):
        if batch_size <= 0 or batch_size > samples.shape[0]:
            batch_size = samples.shape[0]
        return torch.vmap(self.energy_function, chunk_size=batch_size)(samples)
                

    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        print("-" * 100)
        print(f"eval_epoch_end: {prefix}")
        if len(self.eval_step_outputs) == 0:
            print(f"Warning: No eval step outputs for {prefix}")
            return

        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        unprioritized_buffer_samples, cfm_samples, dem_samples = None, None, None
        if self.nll_with_cfm:
            # Compute the negative log likelihood on the buffer samples
            batch_size = (
                len(outputs["data_0"])
                if "data_0" in outputs
                else self.num_buffer_samples_to_generate_per_epoch
            )

            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                batch_size,
                prioritize=self.prioritize_cfm_training_samples,
            )

            cfm_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(batch_size),
            )[-1]

            self.energy_function.log_on_epoch_end(
                latest_samples=self.last_samples,
                latest_energies=self.last_energies,
                wandb_logger=wandb_logger,
                unprioritized_buffer_samples=unprioritized_buffer_samples,
                cfm_samples=cfm_samples,
                replay_buffer=self.buffer,
            )

        else:
            num_samples = (
                len(outputs["data_0"])
                if "data_0" in outputs
                else self.num_buffer_samples_to_generate_per_epoch
            )

            if self.buffer_temperature == "same":
                temperature = self.temperature_schedule(self.global_step)
            else:
                temperature = self.buffer_temperature

            dem_samples = self.generate_samples(
                num_samples=num_samples,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                temperature=temperature,
                batch_size=self.gen_batch_size,
            )
            
            latest_energies = self.get_energies(dem_samples, batch_size=self.gen_batch_size)

            # Only plot dem samples
            self.energy_function.log_on_epoch_end(
                latest_samples=dem_samples,
                latest_energies=latest_energies,
                wandb_logger=wandb_logger,
            )

        if "data_0" in outputs:
            # pad with time dimension 1
            step_samples = cfm_samples if cfm_samples is not None else dem_samples
            samples = step_samples if step_samples is not None else outputs["gen_0"]
            names, dists = compute_distribution_distances(
                self.energy_function.unnormalize(samples)[:, None],
                outputs["data_0"][:, None],
                self.energy_function,
            )
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            self.log_dict(d, sync_dist=True)
        else:
            print(
                f"Warning: No data_0 in outputs for {prefix}. Skipping distribution distances."
            )

        # Compute energy of samples
        try:
            print("=" * 10)
            print(f"assessing samples for {prefix}")
            assessments = self.energy_function.assess_samples(samples)
            if assessments is not None:
                d = {}
                for key, value in assessments.items():
                    d[f"{prefix}/{key}"] = value
                self.log_dict(d, sync_dist=True)
        except Exception as e:
            print(f"Error assessing samples: \n---\n{e}\n---\n")
            print("=" * 10)

        self.eval_step_outputs.clear()

    def _cfm_test_epoch_end(self) -> None:
        """
        Performs end-of-test-epoch operations for the Continuous Flow Matching (CFM) model.

        This method:
        1. Computes negative log-likelihood (NLL) on the full test set
        2. Generates samples from the CFM model
        3. Computes log partition function (log Z) and effective sample size (ESS)
        4. Logs visualizations of the generated samples
        5. Saves the generated samples to disk in two locations:
           - The Hydra output directory
           - A directory named after the energy function
        """
        test_set = self.energy_function.sample_test_set(-1, full=True)
        if test_set is None:
            print("Warning: test_set is None, skipping CFM test epoch end")
            return

        forwards_samples = self.compute_and_log_nll(
            self.cfm_cnf, self.cfm_prior, test_set, "test", ""
        )

        batch_size = self.nll_batch_size
        final_samples = []
        n_batches = math.ceil(self.num_samples_to_save / float(batch_size))
        print(f"Generating {self.num_samples_to_save} CFM samples")
        for i in range(n_batches):
            start = time.time()
            backwards_samples = self.cfm_cnf.generate(
                self.cfm_prior.sample(batch_size),
            )[-1]

            final_samples.append(backwards_samples)
            end = time.time()
            print(f"batch {i} took{end - start: 0.2f}s")
            break

        print("Computing log Z and ESS on generated samples")
        final_samples = torch.cat(final_samples, dim=0)
        assert torch.isfinite(
            final_samples
        ).all(), f"final_samples: Max={final_samples.max()}, Min={final_samples.min()}"

        self.energy_function.log_on_epoch_end(
            latest_samples=final_samples,
            latest_energies=self.energy_function(final_samples),
            wandb_logger=get_wandb_logger(self.loggers),
        )

        self.compute_log_z(self.cfm_cnf, self.cfm_prior, final_samples, "test", "")

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/finalsamples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        # self._log_energy_w2(prefix="test")
        # if self.energy_function.is_molecule:
        #     self._log_dist_w2(prefix="test")
        #     self._log_dist_total_var(prefix="test")

        if self.nll_with_cfm:
            self._cfm_test_epoch_end()
            return

        if self.val_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.val_temperature

        # determine batch size and number of batches
        if self.energy_function.plotting_batch_size > 0:
            batch_size = self.energy_function.plotting_batch_size
            n_batches = self.num_samples_to_save // batch_size
        else:
            # do all in one batch
            batch_size = self.num_samples_to_save
            n_batches = 1
        n_batches = max(n_batches, 1)
        
        final_samples = []
        energies = []
        for i in range(n_batches):
            samples = self.generate_samples(
                num_samples=batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.negative_time,
                temperature=temperature,
            )
            assert torch.isfinite(
                samples
            ).all(), f"samples: Max={samples.max()}, Min={samples.min()}"
            final_samples.append(samples)
            energies.append(self.energy_function(samples).detach())
            
        final_samples = torch.cat(final_samples, dim=0)
        energies = torch.cat(energies, dim=0)
        
        self.energy_function.log_on_epoch_end(
            latest_samples=final_samples,
            latest_energies=energies,
            wandb_logger=wandb_logger,
        )

        if self.energy_function.test_set is not None:
            print("one_test_epoch_end: Computing large batch distribution distances")
            test_set = self.energy_function.sample_test_set(-1, full=True)
            names, dists = compute_full_dataset_distribution_distances(
                self.energy_function.unnormalize(final_samples)[:, None],
                test_set[:, None],
                self.energy_function,
            )
            names = [f"test/full_batch/{name}" for name in names]
            d = dict(zip(names, dists))

            if self.energy_function.is_molecule:
                d["test/full_batch/dist_total_var"] = self._compute_total_var(
                    self.energy_function.unnormalize(final_samples), test_set
                )
            print(
                f"one_test_epoch_end: Done computing large batch distribution distances. W2 = {dists[1]}"
            )
        else:
            print("Warning: No test set provided. Skipping distribution distances.")
            d = {}
            # raise NotImplementedError("No test set provided")

        try:
            assessments = self.energy_function.assess_samples(final_samples)
            if assessments is not None:
                for key, value in assessments.items():
                    d[f"test/{key}"] = value
        except Exception as e:
            print(f"Error assessing samples: \n---\n{e}\n---\n")
            # raise e

        self.log_dict(d, sync_dist=True)

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # insert dem_outputs at the second to last position (should better be done in the config)
        # output_dir = output_dir.split("/")[:-2] + ["dem_outputs"] + output_dir.split("/")[-2:]
        # output_dir = "/".join(output_dir)
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"one_test_epoch_end: Saving samples to {path}")

        path2 = f"dem_outputs/{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        torch.save(final_samples, path2)
        print(f"one_test_epoch_end: Saving samples to {path2}")

    def on_fit_start(self):
        """Called at the beginning of training after sanity check."""
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger:
            self.energy_function.log_datasets(wandb_logger)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """

        # setup energy function
        self.energy_function.setup()

        # # log datasets
        # wandb_logger = get_wandb_logger(self.loggers)
        # self.energy_function.log_datasets(wandb_logger)

        if self.buffer_temperature == "same":
            temperature = self.temperature_schedule(self.global_step)
        else:
            temperature = self.buffer_temperature

        def _grad_fxn(t, x):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                self.noise_schedule,
                self.num_estimator_mc_samples,
                # use_vmap=self.use_vmap,
                temperature=temperature,
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        self.prior = self.partial_prior(
            device=self.device, scale=self.noise_schedule.h(1) ** 0.5
        )
        
        # populate buffer with initial samples
        if self.init_buffer_from_train:
            init_states = self.energy_function.sample_train_set(self.num_buffer_samples_to_generate_init)
        elif self.init_buffer_from_prior:
            init_states = self.prior.sample(self.num_buffer_samples_to_generate_init)
        else:
            init_states = self.generate_samples(
                reverse_sde=reverse_sde,
                num_samples=self.num_buffer_samples_to_generate_init,
                diffusion_scale=self.diffusion_scale,
                temperature=temperature,
                batch_size=self.gen_batch_size,
            )
        init_energies = self.energy_function(init_states)
        self.buffer.add(init_states, init_energies)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm or self.should_train_cfm(0):
            self.cfm_prior = self.partial_prior(
                device=self.device, scale=self.cfm_prior_std
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = DEMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
