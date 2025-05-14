import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.linalg import eigh
import torch

# set seed for reproducibility
np.random.seed(0)

# System parameters
a, b, c = 0.0, -4.0, 0.9
d0 = 1.0
tau = 1.0

def to_center_of_mass_coord(x):
    """
    Compute center of mass coordinates for translation invariance.
    
    Args:
        x: Array of shape (8,) representing four 2D positions concatenated [x1, y1, x2, y2, ...]
        
    Returns:
        Array of shape (8,) with center of mass coordinates
    """
    x = x.reshape(-1, 2)  # Reshape to (4, 2) matrix of positions
    com = np.mean(x, axis=0)  # Compute center of mass
    x_com = x - com  # Subtract center of mass from all positions
    return x_com.flatten()  # Flatten back to (8,) shape

# bgflow/utils/geometry.py
def compute_distances(x, n_particles, n_dimensions, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, n_dimensions)
    distances = torch.cdist(x, x)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)

def energy_loop(x):
    r"""
    Energy of the 4-particle system.
    x has shape (8,) representing four 2D positions concatenated [x1, y1, x2, y2, ...].
    
    DW-4 describes a pair-wise distance potential energy for a system of 4 particles $\{x_1, x_2, x_3, x_4\}$, 
    where each particle has 2 spatial dimensions $x_i \in \mathbb{R}^2$ ($d = 8$). 
    The potential's analytical form is given by:
    $$
    E(x) = \frac{1}{\tau} \sum_{ij} a(d_{ij} - d_0) + b(d_{ij} - d_0)^2 + c(d_{ij} - d_0)^4
    $$
    with 
    $d_{ij} = \|x_i - x_j\|_2$

    We set $a = 0$, $b = -4$, $c = 0.9$, and temperature $\tau = 1$.
    """
    x = x.reshape(-1, 2)
    E = 0.0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dij = np.linalg.norm(x[i] - x[j])
            d = dij - d0
            E += a*d + b*d**2 + c*d**4
    return E / tau

def energy_np(x):
    r"""
    Energy of the 4-particle system using matrix operations.
    x has shape (8,) representing four 2D positions concatenated [x1, y1, x2, y2, ...].
    
    Same potential as energy_loop but implemented with matrix operations.
    """
    x = x.reshape(-1, 2)  # Reshape to (4, 2) matrix of positions
    # Compute all pairwise distances using broadcasting
    diff = x[:, None, :] - x[None, :, :]  # Shape: (4, 4, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (4, 4)
    # Get upper triangle (excluding diagonal) and flatten
    d = dists[np.triu_indices(4, k=1)] - d0
    # Compute energy terms
    E = np.sum(a*d + b*d**2 + c*d**4)
    return E / tau

def grad_energy_np(x):
    """Numerical gradient via finite differences."""
    eps = np.sqrt(np.finfo(float).eps)
    return approx_fprime(x, energy_np, eps)

def hessian_vec_np(x, v):
    """Approximate Hessian-vector product by finite-difference of the gradient."""
    eps = 1e-5
    return (grad_energy_np(x + eps*v) - grad_energy_np(x - eps*v)) / (2*eps)

def energy_torch(x):
    """
    Energy of the 4-particle system using PyTorch.
    x has shape (8,) representing four 2D positions concatenated [x1, y1, x2, y2, ...].
    """
    x = x.reshape(-1, 2)  # Reshape to (4, 2) matrix of positions
    # Compute all pairwise distances using broadcasting
    diff = x[:, None, :] - x[None, :, :]  # Shape: (4, 4, 2)
    dists = torch.sqrt(torch.sum(diff**2, dim=2))  # Shape: (4, 4)
    # Get upper triangle (excluding diagonal) and flatten
    # triu_indices = torch.triu_indices(4, 4, offset=1)
    # d = dists[triu_indices[0], triu_indices[1]] - d0
    d = torch.triu(dists - d0, diagonal=1) 
    # Compute energy terms
    E = torch.sum(a*d + b*d**2 + c*d**4)
    return E / tau

# Convert numpy arrays to torch tensors for comparison
x_np = np.random.randn(8)
x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

# Compute energies
E_np = energy_np(x_np)
E_torch = energy_torch(x_torch)

# Compute gradients
grad_np = grad_energy_np(x_np)
grad_torch = torch.func.grad(energy_torch)(x_torch)

# Compute Hessian-vector products
v = np.random.randn(8)
v_torch = torch.tensor(v, dtype=torch.float32)
hessian_vec_np_val = hessian_vec_np(x_np, v)
hessian_vec_torch_val = torch.func.hessian(energy_torch)(x_torch) 

# Print comparisons
print("Energy comparison:")
print(f"Numpy: {E_np:.6f}")
print(f"PyTorch: {E_torch.item():.6f}")
print(f"Difference: {abs(E_np - E_torch.item()):.2e}")

print("\nGradient comparison:")
print(f"Numpy: {grad_np}")
print(f"PyTorch: {grad_torch.detach().numpy()}")
print(f"Max difference: {np.max(np.abs(grad_np - grad_torch.detach().numpy())):.2e}")

print("\nHessian-vector product comparison:")
print(f"Numpy: {hessian_vec_np_val}")
print(f"PyTorch: {hessian_vec_torch_val.detach().numpy()}")
print(f"Max difference: {np.max(np.abs(hessian_vec_np_val - hessian_vec_torch_val.detach().numpy())):.2e}")
