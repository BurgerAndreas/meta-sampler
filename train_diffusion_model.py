import torch
import numpy as np
import matplotlib.pyplot as plt

# Define a 1D tensor for x over an interval; here we use 400 points in the range [-3, 3]
x = torch.linspace(-3, 3, 400, requires_grad=True)

# Define the test function: f(x) = (x^2 - 1)^3.
# This function has:
#   - A local minimum at x = 0 (f(0) = -1)
#   - Degenerate (inflection/saddle) points at x = -1 and x = 1 (f(±1) = 0)
f = (x**2 - 1)**3

# Compute the derivative using PyTorch's autograd.
grad_f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

# Convert the torch tensors to NumPy arrays for plotting.
x_np = x.detach().numpy()
f_np = f.detach().numpy()
grad_f_np = grad_f.detach().numpy()

# Compute the Boltzmann distribution using the absolute derivative as the energy.
# Set beta = 1.0 (you can adjust this as needed)
beta = 1.0
# Energy E(x) is defined as |f'(x)|
E = np.abs(grad_f_np)
# Compute the Boltzmann factor e^(-beta * E)
boltz_factor = np.exp(-beta * E)
# Normalize the distribution using the trapezoidal rule to approximate the integral
Z = np.trapz(boltz_factor, x_np)
# Boltzmann probability distribution
p_x = boltz_factor / Z

# Create a figure with 3 subplots: one each for f(x), |f'(x)|, and the Boltzmann distribution p(x)
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot the function f(x)
axs[0].plot(x_np, f_np, label=r'$f(x) = (x^2-1)^3$', color='blue')
# Mark saddle points at x = -1 and x = 1
axs[0].axvline(x=-1, color='red', linestyle='--', label='Saddle Points')
axs[0].axvline(x=1, color='red', linestyle='--')
axs[0].set_ylabel('f(x)')
axs[0].set_title('Test Function f(x)')
axs[0].legend(loc='upper left')
axs[0].grid(True)
axs[0].set_ylim(-5, 60)

# Plot the absolute derivative |f'(x)|
axs[1].plot(x_np, np.abs(grad_f_np), label=r"$|f'(x)|$", color='green')
axs[1].set_ylabel(r"|f'(x)|")
axs[1].set_title('Absolute Derivative |f\'(x)|')
axs[1].legend(loc='upper left')
axs[1].grid(True)
axs[1].set_ylim(-10, 10)

# Plot the Boltzmann distribution p(x)
axs[2].plot(x_np, p_x, label=r'Boltzmann Distribution', color='purple')
axs[2].set_xlabel('x')
axs[2].set_ylabel('p(x)')
axs[2].set_title(r'Boltzmann Distribution with Energy $|f\'(x)|$')
axs[2].legend(loc='upper left')
axs[2].grid(True)

plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

###############################################
# 1. Define the function, its derivative, and the Boltzmann distribution
###############################################
# Create a 1D tensor for x over an interval; here we use 400 points in the range [-3, 3]
x = torch.linspace(-3, 3, 400, requires_grad=True)

# Define the test function f(x)
# f(x) = (x^2 - 1)^3
f = (x**2 - 1)**3

# Compute the derivative using PyTorch's autograd.
grad_f = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]

# Convert the torch tensors to NumPy arrays for later use.
x_np = x.detach().numpy()
f_np = f.detach().numpy()
grad_f_np = grad_f.detach().numpy()

# Build the Boltzmann distribution using the absolute derivative as energy.
beta_energy = 1.0  # inverse temperature for the Boltzmann factor
E = np.abs(grad_f_np)  # energy is given by |f'(x)|
boltz_factor = np.exp(-beta_energy * E)
# Normalize using the trapezoidal rule (for plotting on a grid)
Z = np.trapz(boltz_factor, x_np)
p_x = boltz_factor / Z  # target probability density on the grid

###############################################
# 2. Create Training Samples from the Boltzmann Distribution
###############################################
# We'll generate training data by sampling from the discrete grid weighted by the Boltzmann distribution.
# (We normalize the discrete weights to sum to one)
p_grid = p_x / np.sum(p_x)
n_train = 10000  # number of training samples
train_samples_np = np.random.choice(x_np, size=n_train, p=p_grid)
# Convert the training samples to a torch tensor of shape (n_train, 1)
train_samples = torch.tensor(train_samples_np, dtype=torch.float32).unsqueeze(1)

###############################################
# 3. Define the Diffusion Model Parameters and Helper Functions
###############################################
# Diffusion hyperparameters
T = 100            # total diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)  # shape (T,)
alphas = 1 - betas                              # shape (T,)
alpha_bar = torch.cumprod(alphas, dim=0)         # cumulative product

# We'll be using the same device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = betas.to(device)
alphas = alphas.to(device)
alpha_bar = alpha_bar.to(device)

###############################################
# 4. Define a Very Small Neural Network to Predict the Noise
###############################################
class DiffusionNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super(DiffusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, t):
        # x: tensor of shape (batch, 1)
        # t: tensor of shape (batch, 1) -- we will pass normalized timestep (between 0 and 1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

# Instantiate the model, move to device, and create an optimizer.
model = DiffusionNet(hidden_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

###############################################
# 5. Training Loop for the Diffusion Model
###############################################
num_epochs = 3000
batch_size = 128

# To simplify, we will not use DataLoader; instead, we randomly sample minibatches.
n_batches = n_train // batch_size

print("Starting training...")
for epoch in range(num_epochs):
    perm = torch.randperm(n_train)
    epoch_loss = 0.0
    for i in range(n_batches):
        optimizer.zero_grad()
        indices = perm[i * batch_size:(i + 1) * batch_size]
        x0 = train_samples[indices].to(device)  # clean sample from the Boltzmann distribution, shape: (batch, 1)

        # Sample a diffusion step t for each data point uniformly from {0,..., T-1}
        t = torch.randint(0, T, (batch_size,), device=device)  # integer timesteps
        # Normalize time to be between 0 and 1 for input to the network
        t_norm = (t.float() / T).unsqueeze(1)  # shape: (batch, 1)

        # Get the corresponding alpha_bar for each t and reshape for broadcasting
        alpha_bar_t = alpha_bar[t].unsqueeze(1)  # shape: (batch, 1)

        # Sample noise: epsilon ~ N(0, 1)
        noise = torch.randn_like(x0)

        # Create the noisy version of x0 from the forward-diffusion process:
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_bar = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt()
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        # The model will try to predict the noise added given (x_t, t_norm)
        pred_noise = model(x_t, t_norm)
        loss = mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/n_batches:.5f}")

###############################################
# 6. Reverse Diffusion Sampling (Generate New Samples)
###############################################
# We'll generate new samples by starting from Gaussian noise and running the reverse process.
n_gen = 1000  # number of samples to generate
# Initialize x_T ~ N(0,1)
x_gen = torch.randn(n_gen, 1, device=device)

# Reverse diffusion: iterate t=T-1...0
for t_inv in range(T-1, -1, -1):
    t_batch = (torch.ones(n_gen, 1) * (t_inv / T)).to(device)  # normalized t (scalar repeated for batch)
    # Get the diffusion parameters for the current timestep.
    beta_t = betas[t_inv]           # scalar
    alpha_t = alphas[t_inv]           # scalar
    alpha_bar_t = alpha_bar[t_inv]    # scalar

    # Predict the noise using our network.
    with torch.no_grad():
        pred_noise = model(x_gen, t_batch)  # shape (n_gen, 1)

    # Compute the reverse update:
    # x_{t-1} = 1/sqrt(alpha_t)*( x_t - (beta_t/ sqrt(1 - alpha_bar_t))*pred_noise )
    coeff = beta_t / ( (1 - alpha_bar_t).sqrt() )
    x_gen = (1 / (alpha_t**0.5)) * (x_gen - coeff * pred_noise)
    
    # Add noise when t > 0
    if t_inv > 0:
        noise = torch.randn_like(x_gen)
        sigma_t = beta_t.sqrt()
        x_gen = x_gen + sigma_t * noise

# Bring generated samples to CPU and convert to numpy.
x_gen_np = x_gen.squeeze().cpu().numpy()

###############################################
# 7. Plotting the Results: True Distribution vs. Generated Samples
###############################################
fig, ax = plt.subplots(figsize=(10, 6))
# Plot histogram of generated samples.
ax.hist(x_gen_np, bins=50, density=True, alpha=0.6, label="Generated Samples")

# Overlay the true Boltzmann distribution.
ax.plot(x_np, p_x, 'r-', lw=2, label="Target Boltzmann Distribution")
ax.set_title("Diffusion Sampler: Generated Samples vs Target Distribution")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
plt.show()