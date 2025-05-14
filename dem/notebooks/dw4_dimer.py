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

def energy(x):
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

def grad_energy(x):
    """Numerical gradient via finite differences."""
    eps = np.sqrt(np.finfo(float).eps)
    return approx_fprime(x, energy, eps)

def hessian_vec(x, v):
    """Approximate Hessian-vector product by finite-difference of the gradient."""
    eps = 1e-5
    return (grad_energy(x + eps*v) - grad_energy(x - eps*v)) / (2*eps)

# 1. Find two minima starting from random initial guesses
minima = []
for seed in range(10):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(8) * 0.5  # small random initial positions
    res = minimize(energy, x0, jac=grad_energy, method='L-BFGS-B')
    # check if this minimum is unique compared to already found ones
    is_unique = True
    for x_unique in minima:
        if np.linalg.norm(res.x - x_unique) < 1e-6:
            is_unique = False
            break
    if is_unique:
        minima.append(res.x)
    print(f"Minimum {len(minima)}: E = {res.fun:.6f}, success = {res.success}")

assert len(minima) >= 2, f"Found {len(minima)} minima"
xA, xB = minima[0], minima[1]

# 2. Dimer method to locate a saddle between xA and xB
def dimer_method(x_init, max_iter=200, tol=1e-5, step_size=1e-1, dimer_dist=1e-3):
    x = x_init.copy()
    # Initialize random orientation vector v
    v = np.random.randn(len(x))
    v /= np.linalg.norm(v)
    
    for iteration in range(max_iter):
        # construct dimer images
        xp = x + dimer_dist * v
        xm = x - dimer_dist * v
        
        # compute gradients at the images
        gp = grad_energy(xp)
        gm = grad_energy(xm)
        
        # compute rotational part: torque
        torque = (gp - gm) / (2 * dimer_dist) - np.dot(v, (gp - gm) / (2 * dimer_dist)) * v
        
        # update direction v to minimize curvature
        v -= step_size * torque
        v /= np.linalg.norm(v)
        
        # compute effective force
        g0 = grad_energy(x)
        F_eff = g0 - 2 * np.dot(g0, v) * v
        
        # translate step
        x -= step_size * F_eff
        
        # convergence check on gradient norm
        if np.linalg.norm(g0) < tol:
            print(f"Dimer converged in {iteration+1} iters, |grad|={np.linalg.norm(g0):.2e}")
            break
    return x

# initial guess: midpoint of minima
x0_dimer = 0.5*(xA + xB)
x_saddle = dimer_method(x0_dimer)

# 3. Polish saddle with a few Newton steps using approximate Hessian
for _ in range(5):
    g = grad_energy(x_saddle)
    # approximate Hessian matrix via finite differences
    H = np.zeros((8,8))
    basis = np.eye(8)
    for i in range(8):
        H[:,i] = (grad_energy(x_saddle + 1e-5*basis[i]) - grad_energy(x_saddle - 1e-5*basis[i]))/(2e-5)
    # solve for step
    try:
        dx = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        break
    x_saddle += dx

# 4. Verify saddle: gradient approx zero and Hessian has one negative eigenvalue
grad_norm = np.linalg.norm(grad_energy(x_saddle))
eigs = eigh(H, eigvals_only=True)
num_negative = np.sum(eigs < 0)

print(f"\nPolished saddle: |grad| = {grad_norm:.2e}")
print(f"Hessian eigenvalues: {eigs}")
print(f"Number of negative eigenvalues: {num_negative}")
