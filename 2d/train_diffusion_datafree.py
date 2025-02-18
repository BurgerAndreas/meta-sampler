import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import time
import hydra
from omegaconf import DictConfig
import os
import pathlib

from toy_functions import toy_polynomial_2d, six_hump_camel
from plotting import plot_potential_energy_surface, plot_results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_diffusion_model(cfg: DictConfig):
    """
    Train a diffusion model to match the pseudo_energy of a given potential_energy_surface.

    Args:
        cfg (DictConfig): Hydra config containing model parameters and hyperparameters

    Returns:
        None
    """
    # ---------------------------
    # Diffusion Hyperparameters
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = cfg.diffusion.timesteps  # number of diffusion steps
    beta_start = cfg.diffusion.beta_start
    beta_end = cfg.diffusion.beta_end
    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    pes_fn = eval(cfg.pes_fn)

    plot_potential_energy_surface(
        pes_fn, cutoff=cfg.plotting.cutoff, percentile=cfg.plotting.percentile
    )

    # ---------------------------
    # Define the Diffusion Model (a small neural network for 2D data)
    # ---------------------------
    class DiffusionNet(nn.Module):
        def __init__(self, hidden_dim=32):
            super(DiffusionNet, self).__init__()
            # Input now comprises 2 coordinates plus 1 time dimension = 3.
            self.net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # Output: predicted noise for both x and y.
            )

        def forward(self, x, t):
            # x: tensor of shape (batch, 2)
            # t: tensor of shape (batch, 1), normalized timestep.
            inp = torch.cat([x, t], dim=1)
            return self.net(inp)

    # ---------------------------
    # Setup: device, model, optimizer, etc.
    # ---------------------------
    model = DiffusionNet(hidden_dim=cfg.model.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    mse_loss = nn.MSELoss()

    # Move diffusion hyperparameters to the same device.
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    # ---------------------------
    # Iterative Diffusion pseudo_energy Matching: Training without target Boltzmann samples.
    # ---------------------------
    outer_iterations = (
        cfg.training.outer_iterations
    )  # number of outer loops updating the pseudo-data
    inner_epochs = (
        cfg.training.inner_epochs
    )  # inner training epochs per outer iteration
    batch_size = cfg.training.batch_size
    lambda_pseudo_energy = (
        cfg.training.lambda_pseudo_energy
    )  # weight for the pseudo_energy penalty

    n_samples = cfg.training.n_samples  # number of pseudo-data points

    # Initialize pseudo-data: sample uniformly from [-3, 3] in 2D.
    with torch.no_grad():
        samples_buffer = torch.FloatTensor(n_samples, 2).uniform_(-3, 3).to(device)

    print("\nStarting iDEM training...")
    for outer in range(outer_iterations):
        # INNER TRAINING LOOP: update the model using the current pseudo-data.
        n_batches = n_samples // batch_size
        for epoch in range(inner_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(n_batches):
                optimizer.zero_grad()
                indices = perm[i * batch_size : (i + 1) * batch_size]
                x0 = samples_buffer[indices]  # pseudo-data, shape (batch, 2)

                # Sample a random diffusion timestep t for each data point.
                t = torch.randint(0, T, (batch_size,), device=device)
                t_norm = (t.float() / T).unsqueeze(1)  # normalized timestep.

                # Get the corresponding alpha_bar for each t.
                alpha_bar_t = alpha_bar[t].unsqueeze(1)
                sqrt_alpha_bar = alpha_bar_t.sqrt()
                sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()

                # Sample noise ~ N(0,1), matching the shape of x0.
                noise = torch.randn_like(x0)

                # Forward diffusion: simulate x_t from the pseudo-data x0.
                x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

                # Predict the noise using the diffusion network.
                pred_noise = model(x_t, t_norm)

                # Standard diffusion (denoising) loss.
                loss_diff = mse_loss(pred_noise, noise)

                # Estimate the reconstructed clean sample x0_pred.
                x0_pred = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
                # pseudo_energy loss: encourage x0_pred to have low pseudo_energy.
                pes, pes_grad_norm = pes_fn(x0_pred)
                loss_pseudo_energy = pes_grad_norm.mean()
                loss = loss_diff + lambda_pseudo_energy * loss_pseudo_energy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 100 == 0:
                print(
                    f"Outer iter {outer+1}/{outer_iterations}, Epoch [{epoch+1}/{inner_epochs}] Loss: {epoch_loss/n_batches:.5f}"
                )

        # AFTER INNER TRAINING: update pseudo-data using reverse diffusion (sampling).
        with torch.no_grad():
            # Start from Gaussian noise at time T.
            x_gen = torch.randn(n_samples, 2, device=device)
            # Reverse diffusion process from t = T-1 down to 0.
            for t_inv in range(T - 1, -1, -1):
                t_batch = (torch.ones(n_samples, 1) * (t_inv / T)).to(
                    device
                )  # normalized time.
                beta_t = betas[t_inv]
                alpha_t = alphas[t_inv]
                alpha_bar_t = alpha_bar[t_inv]

                # Predict noise from the model.
                pred_noise = model(x_gen, t_batch)

                # Reverse update step (following standard DDPM update).
                coeff = beta_t / ((1 - alpha_bar_t).sqrt())
                x_gen = (1 / (alpha_t**0.5)) * (x_gen - coeff * pred_noise)

                # Add noise for t > 0.
                if t_inv > 0:
                    noise = torch.randn_like(x_gen)
                    sigma_t = beta_t.sqrt()
                    x_gen = x_gen + sigma_t * noise
            # Update the pseudo-data buffer.
            samples_buffer = x_gen.detach()

        # Optionally: print the average pseudo_energy of the current pseudo-data.
        pes, pes_grad_norm = pes_fn(samples_buffer)
        avg_pes = pes.mean().item()
        avg_pes_grad_norm = pes_grad_norm.mean().item()
        print(
            f"{outer+1}: Avg(PES) = {avg_pes:.5f}, Avg(|∇PES|) = {avg_pes_grad_norm:.5f}"
        )

    # ---------------------------
    # 6. Final Sampling and Comparison to Target Boltzmann Distribution
    # ---------------------------
    with torch.no_grad():
        n_gen = cfg.sampling.n_samples
        x_gen = torch.randn(n_gen, 2, device=device)
        for t_inv in range(T - 1, -1, -1):
            t_batch = (torch.ones(n_gen, 1) * (t_inv / T)).to(device)
            beta_t = betas[t_inv]
            alpha_t = alphas[t_inv]
            alpha_bar_t = alpha_bar[t_inv]

            pred_noise = model(x_gen, t_batch)
            coeff = beta_t / ((1 - alpha_bar_t).sqrt())
            x_gen = (1 / (alpha_t**0.5)) * (x_gen - coeff * pred_noise)
            if t_inv > 0:
                noise = torch.randn_like(x_gen)
                sigma_t = beta_t.sqrt()
                x_gen = x_gen + sigma_t * noise
        final_samples = x_gen.squeeze().cpu().numpy()

    plot_results(final_samples)


if __name__ == "__main__":
    train_diffusion_model()
