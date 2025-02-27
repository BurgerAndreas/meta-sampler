import torch
from fab.target_distributions.gmm import GMM

# Initialize GMM from fab
gmm = GMM(
    dim=2,
    n_mixes=40, 
    loc_scaling=40,
    log_var_scaling=1.0,
    use_gpu=False,
)

# Get centers and variances from torch GMM
centers = gmm.locs.detach().cpu().numpy()
variances = gmm.scale_trils.detach().cpu().numpy()

print(f"centers: \n{centers}")
print(f"\nvariances: \n{variances}")

# import jax
# import jax.numpy as jnp

# # Build identical GMM in JAX
# class JAXGMM:
#     def __init__(self, centers, variances):
#         self.centers = jnp.array(centers)
#         self.variances = jnp.array(variances)
#         self.n_components = len(centers)
#         self.weights = jnp.ones(self.n_components) / self.n_components
        
#     def log_prob(self, x):
#         # Compute log probability for each component
#         diff = x[:, None, :] - self.centers[None, :, :]
#         log_probs = -0.5 * jnp.sum(diff * jnp.linalg.solve(self.variances, diff[..., None])[..., 0], axis=-1)
#         log_probs = log_probs - 0.5 * jnp.log(jnp.linalg.det(self.variances)) - self.centers.shape[1] * jnp.log(2 * jnp.pi) / 2
        
#         # Add log weights and use logsumexp for numerical stability
#         log_probs = log_probs + jnp.log(self.weights)
#         return jax.scipy.special.logsumexp(log_probs, axis=1)

# # Create JAX GMM instance
# jax_gmm = JAXGMM(centers, variances)

# # Test log_prob function
# test_points = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
# print(jax_gmm.log_prob(test_points))
# print(gmm.log_prob(test_points))