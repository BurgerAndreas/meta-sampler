import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# plotting directory: current directory
plotting_dir = pathlib.Path(__file__).parent / "plots"
plotting_dir.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 1. Define our toy function and its pseudo_energy.
# ---------------------------
def pes_fn(x):
    # f(x) = (x**2 - 1)**3
    return (x**2 - 1) ** 3


def f_prime(x):
    # Analytical derivative: f'(x)=6*x*(x**2-1)**2
    return 6 * x * (x**2 - 1) ** 2


def pseudo_energy(x):
    # pseudo_energy defined as the absolute value of f'(x)
    # return torch.abs(f_prime(x))
    return torch.linalg.norm(f_prime(x), dim=1)


if __name__ == "__main__":
    # Define a 1D tensor for x over an interval; here we use 400 points in the range [-3, 3]
    xvals = torch.linspace(-3, 3, 400, requires_grad=True)

    # Define the test function: f(x) = (x^2 - 1)^3.
    # This function has:
    #   - A local minimum at x = 0 (f(0) = -1)
    #   - Degenerate (inflection/saddle) points at x = -1 and x = 1 (f(±1) = 0)
    pes_val = pes_fn(xvals)

    # Compute the derivative using PyTorch's autograd.
    grad_f = torch.autograd.grad(
        pes_val, xvals, grad_outputs=torch.ones_like(pes_val), create_graph=True
    )[0]

    # Convert the torch tensors to NumPy arrays for plotting.
    x_np = xvals.detach().numpy()
    f_np = pes_val.detach().numpy()
    grad_f_np = grad_f.detach().numpy()

    # compare autograd and analytical gradient
    print("grad_f_np", grad_f_np.shape)
    grad_f_np_analytical = f_prime(xvals).detach().numpy()
    print("grad_f_np_analytical", grad_f_np_analytical.shape)
    print("diff", np.abs(grad_f_np - grad_f_np_analytical).mean())
    # in 1d, linalg.norm is the same as abs
    print(
        "diff", (np.abs(grad_f_np) - np.linalg.norm(grad_f_np[:, None], axis=1)).mean()
    )

    # Compute the Boltzmann distribution using the absolute derivative as the pseudo_energy.
    # Set beta = 1.0 (you can adjust this as needed)
    beta = 1.0
    # pseudo_energy E(x) is defined as |f'(x)|
    E = np.abs(grad_f_np)
    # Compute the Boltzmann factor e^(-beta * E)
    boltz_factor = np.exp(-beta * E)
    # Normalize the distribution using the trapezoidal rule to approximate the integral
    Z = np.trapezoid(boltz_factor, x_np)
    # Boltzmann probability distribution
    p_x = boltz_factor / Z

    # Create a figure with 3 subplots: one each for f(x), |f'(x)|, and the Boltzmann distribution p(x)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot the function f(x)
    axs[0].plot(x_np, f_np, label=r"$f(x) = (x^2-1)^3$", color="blue")
    # Mark saddle points at x = -1 and x = 1
    axs[0].axvline(x=-1, color="red", linestyle="--", label="Saddle Points")
    axs[0].axvline(x=1, color="red", linestyle="--")
    axs[0].set_ylabel("f(x)")
    axs[0].set_title("Test Function f(x)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True)
    axs[0].set_ylim(-2, 5)

    # Plot the absolute derivative |f'(x)|
    axs[1].plot(x_np, np.abs(grad_f_np), label=r"$|f'(x)|$", color="green")
    axs[1].set_ylabel(r"|f'(x)|")
    axs[1].set_title("Absolute Derivative |f'(x)|")
    axs[1].legend(loc="upper left")
    axs[1].grid(True)
    axs[1].set_ylim(-10, 10)

    # Plot the Boltzmann distribution p(x)
    axs[2].plot(x_np, p_x, label=r"Boltzmann Distribution", color="purple")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("p(x)")
    axs[2].set_title(r"Boltzmann Distribution with pseudo_energy $|f\'(x)|$")
    axs[2].legend(loc="upper left")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(plotting_dir / "diffusion_datafree_1d_target_distribution.png")
    print(
        "Saved figure to ",
        plotting_dir / "diffusion_datafree_1d_target_distribution.png",
    )

    # ---------------------------
    # 2. Diffusion Hyperparameters
    # ---------------------------
    T = 100  # number of diffusion steps
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # Hyperparameters for the iterative procedure:
    outer_iterations = 5  # number of outer loops updating the pseudo-data
    inner_epochs = 300  # inner training epochs per outer iteration
    batch_size = 128
    lambda_pseudo_energy = 0.1  # weight for the pseudo_energy penalty
    n_samples = 10000  # number of pseudo-data points (Training set size)

    # ---------------------------
    # 3. Define the Diffusion Model (a small neural network)
    # ---------------------------
    class DiffusionNet(nn.Module):
        def __init__(self, hidden_dim=32):
            super(DiffusionNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, t):
            # x: tensor of shape (batch, 1)
            # t: tensor of shape (batch, 1) (normalized timestep between 0 and 1)
            inp = torch.cat([x, t], dim=1)
            return self.net(inp)

    # ---------------------------
    # 4. Setup: device, model, optimizer, etc.
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionNet(hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # IMPORTANT: Move diffusion hyperparameters to the same device.
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    # ---------------------------
    # 5. Iterative Diffusion pseudo_energy Matching: Training without target Boltzmann samples.
    # ---------------------------

    # Initialize pseudo-data: sample uniformly from [-3, 3]
    with torch.no_grad():
        samples_buffer = torch.FloatTensor(n_samples, 1).uniform_(-3, 3).to(device)

    print("\nStarting iterative diffusion pseudo_energy matching training...")
    for outer in range(outer_iterations):
        # INNER TRAINING LOOP: update the model using the current sample buffer (no ground-truth data)
        n_batches = n_samples // batch_size
        for epoch in range(inner_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(n_batches):
                optimizer.zero_grad()
                indices = perm[i * batch_size : (i + 1) * batch_size]
                x0 = samples_buffer[indices]  # pseudo-data, shape (batch,1)

                # Sample a random diffusion timestep t for each data point (make sure it's on device)
                t = torch.randint(0, T, (batch_size,), device=device)
                t_norm = (t.float() / T).unsqueeze(
                    1
                )  # normalized timestep, shape (batch,1)

                # Get corresponding alpha_bar for each t. Now, since alpha_bar is on device, this will work.
                alpha_bar_t = alpha_bar[t].unsqueeze(1)
                sqrt_alpha_bar = alpha_bar_t.sqrt()
                sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()

                # Sample noise epsilon ~ N(0,1)
                noise = torch.randn_like(x0)

                # Forward diffusion: simulate x_t from the pseudo-data x0
                x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

                # Predict the noise using the current diffusion network
                pred_noise = model(x_t, t_norm)

                # Standard diffusion (denoising) loss:
                loss_diff = mse_loss(pred_noise, noise)

                # Estimate the predicted clean sample x0 from x_t and pred_noise:
                x0_pred = (x_t - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
                # pseudo_energy loss: encourage x0_pred to have low pseudo_energy
                loss_pseudo_energy = pseudo_energy(x0_pred).mean()

                loss = loss_diff + lambda_pseudo_energy * loss_pseudo_energy
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 100 == 0:
                print(
                    f"Outer iter {outer+1}/{outer_iterations}, Epoch [{epoch+1}/{inner_epochs}] Loss: {epoch_loss/n_batches:.5f}"
                )

        # AFTER INNER TRAINING: update pseudo-data using reverse diffusion (sampling)
        with torch.no_grad():
            # Start from Gaussian noise at time T.
            x_gen = torch.randn(n_samples, 1, device=device)
            # Reverse diffusion process from t=T-1 down to 0
            for t_inv in range(T - 1, -1, -1):
                t_batch = (torch.ones(n_samples, 1) * (t_inv / T)).to(
                    device
                )  # normalized time
                beta_t = betas[t_inv]
                alpha_t = alphas[t_inv]
                alpha_bar_t = alpha_bar[t_inv]

                # Predict noise from the model.
                pred_noise = model(x_gen, t_batch)

                # Reverse update (following standard DDPM update)
                coeff = beta_t / ((1 - alpha_bar_t).sqrt())
                x_gen = (1 / (alpha_t**0.5)) * (x_gen - coeff * pred_noise)

                # Add noise for t > 0
                if t_inv > 0:
                    noise = torch.randn_like(x_gen)
                    sigma_t = beta_t.sqrt()
                    x_gen = x_gen + sigma_t * noise
            # Update the sample buffer with the new samples for the next iteration.
            samples_buffer = x_gen.detach()

        # Optionally: print average pseudo_energy of the current pseudo-data.
        avg_pseudo_energy = pseudo_energy(samples_buffer).mean().item()
        print(
            f"After outer iteration {outer+1}, average pseudo_energy of pseudo-data: {avg_pseudo_energy:.5f}"
        )

    # ---------------------------
    # 6. Final Sampling and Comparison to Target Boltzmann Distribution
    # ---------------------------
    # Generate final samples by running reverse diffusion one more time.
    with torch.no_grad():
        n_gen = 10000
        x_gen = torch.randn(n_gen, 1, device=device)
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

    # Compute the true Boltzmann density over a grid:
    x_grid = np.linspace(-3, 3, 400)
    # Compute pseudo_energy on grid (using numpy version of f_prime)
    fprime_grid = 6 * x_grid * (x_grid**2 - 1) ** 2
    U_grid = np.abs(fprime_grid)
    p_true = np.exp(-U_grid)
    # Normalize via trapezoidal integration
    p_true = p_true / np.trapezoid(p_true, x_grid)

    # ---------------------------
    # 7. Plot the Results
    # ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        final_samples,
        bins=50,
        density=True,
        alpha=0.6,
        label="Generated Samples (iDEM)",
    )
    ax.plot(x_grid, p_true, "r-", lw=2, label="Target Boltzmann Distribution")
    ax.set_title(
        "Iterative Diffusion pseudo_energy Matching for Sampling\nfrom the Boltzmann Distribution"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(plotting_dir / "diffusion_datafree_1d.png")
    print("Saved figure to ", plotting_dir / "diffusion_datafree_1d.png")
