import jax
import numpy as np

import jax.numpy as jnp
from jax import random

import flax

import optax

from flax import linen as nn
from flax.training import train_state

from tqdm import trange
from functools import partial
from matplotlib import pyplot as plt
from typing import NamedTuple, Any
import os

plotfolder = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(plotfolder, exist_ok=True)

# Data Generation

"""
This file implements diffusion models in JAX and Flax on a simple 2D toy dataset.
It demonstrates various methods of combining multiple diffusion models:
- Standard sampling from individual models
- Isosurface sampling (equal density sampling)
- Simple 50/50 averaging of model outputs
- Density-proportional averaging
- Stochastic superposition
"""


def sample_data(key, bs, up=True):
    """
    Generate synthetic 2D data points.

    Args:
      key: JAX random key
      bs: Batch size
      up: If True, generate data in the upper region, otherwise in the lower region

    Returns:
      2D data points of shape (bs, 2)
    """
    keys = random.split(key, 3)
    if up:
        x_1 = random.randint(
            keys[0], minval=jnp.array([0, 1]), maxval=jnp.array([2, 2]), shape=(bs, 2)
        )
    else:
        x_1 = random.randint(
            keys[0], minval=jnp.array([0, 0]), maxval=jnp.array([2, 1]), shape=(bs, 2)
        )
    x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
    x_1 += 4e-1 * random.normal(keys[1], shape=(bs, 2))
    return x_1


# Diffusion model parameters
ndim = 2
t_0, t_1 = 0.0, 1.0
beta_0 = 0.1
beta_1 = 20.0


# Define noise schedule functions
def log_alpha(t):
    return -0.5 * t * beta_0 - 0.25 * t**2 * (beta_1 - beta_0)


def log_sigma(t):
    # log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
    return jnp.log(t)


dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())


def beta(t):
    # beta_t = s_t d/dt log(s_t/alpha_t)
    # beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
    return 1 + 0.5 * t * beta_0 + 0.5 * t**2 * (beta_1 - beta_0)


def q_t(key, data, t):
    """
    Forward diffusion process: add noise to data according to time t.

    Args:
      key: JAX random key
      data: Input data points
      t: Time parameter (0 = no noise, 1 = pure noise)

    Returns:
      Tuple of (noise, noisy_data)
    """
    eps = random.normal(key, shape=data.shape)
    x_t = jnp.exp(log_alpha(t)) * data + jnp.exp(log_sigma(t)) * eps
    return eps, x_t


# Set random seed for reproducibility
seed = 0
np.random.seed(seed)
key = random.PRNGKey(seed)
bs = 512
t_axis = np.linspace(0.0, 1.0, 6)

# Visualize the forward diffusion process
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 5)
    _, x_t_up = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t_axis[i])
    _, x_t_down = q_t(ikey[3], sample_data(ikey[2], bs // 2, up=False), t_axis[i])
    plt.scatter(x_t_up[:, 0], x_t_up[:, 1], alpha=0.3)
    plt.scatter(x_t_down[:, 0], x_t_down[:, 1], alpha=0.3)
    plt.title(f"t={t_axis[i]}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
fname = os.path.join(plotfolder, "toy-data.png")
plt.savefig(fname)
print(f"Saved {fname}")

# Define the Model


class MLP(nn.Module):
    """
    Multi-layer perceptron model for score estimation.

    Attributes:
      num_hid: Number of hidden units in each layer
      num_out: Number of output dimensions
    """

    num_hid: int
    num_out: int

    @nn.compact
    def __call__(self, t, x):
        """
        Forward pass of the MLP.

        Args:
          t: Time parameter of shape (batch_size, 1)
          x: Input data of shape (batch_size, ndim)

        Returns:
          Output of shape (batch_size, num_out)
        """
        h = jnp.hstack([t, x])
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_out)(h)
        return h


def train_model(key, data_generator):
    """
    Train a diffusion model.

    Args:
      key: JAX random key
      data_generator: Function that generates training data

    Returns:
      Trained model state
    """
    model = MLP(num_hid=512, num_out=ndim)
    key, init_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, np.ones([bs, 1]), np.zeros([bs, ndim])),
        tx=optimizer,
    )

    def sm_loss(state, key, params, bs):
        """Score matching loss function"""
        keys = random.split(
            key,
        )

        def sdlogqdx(_t, _x):
            return state.apply_fn(params, _t, _x)

        data = data_generator(keys[0], bs)
        t = random.uniform(keys[1], [bs, 1])
        eps, x_t = q_t(keys[2], data, t)
        loss = ((eps + sdlogqdx(t, x_t)) ** 2).sum(1)
        return loss.mean()

    @partial(jax.jit, static_argnums=1)
    def train_step(state, bs, key):
        """Single training step with gradient update"""
        grad_fn = jax.value_and_grad(sm_loss, argnums=2)
        loss, grads = grad_fn(state, key, state.params, bs)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Training loop
    num_iterations = 20_000
    key, loop_key = random.split(key)
    for iter in trange(num_iterations):
        state, _ = train_step(state, bs, random.fold_in(loop_key, iter))
    return state


# Train two models: one for "up" data and one for "down" data
key, ikey = random.split(key)
state_up = train_model(ikey, partial(sample_data, up=True))
key, ikey = random.split(key)
state_down = train_model(ikey, partial(sample_data, up=False))


# Evaluation of the Trained Model


# Vector field for SDE sampling
# v_t(x) = dlog(alpha)/dt x - s^2_t d/dt log(s_t/alpha_t) dlog q_t(x)/dx
# beta_t = s^2_t d/dt log(s_t/alpha_t)
# dx = (v_t(x) - xi*beta_t*dlog q_t(x)/dx)dt + sqrt(2*beta_t*xi*dt)*eps
# drift f_t(x) = v_t(x) + eps_t * dlog(q_t(x))/dx
@jax.jit
def vector_field_SDE(state, t, x):
    """
    Compute the vector field for SDE sampling.

    Args:
      state: Model state
      t: Time parameter
      x: Input data

    Returns:
      Vector field at (t,x)
    """

    def sdlogqdx(_t, _x):
        return state.apply_fn(state.params, _t, _x)

    dxdt = dlog_alphadt(t) * x - 2 * beta(t) * sdlogqdx(t, x)
    return dxdt

@jax.jit
def score_and_divergence(key, t, x, state):
    r"""
    Compute the vector field and its divergence using JVP.
    
    Hutchinson trace estimator
    $$
    \text{Tr}(\nabla_x \mathbf{f}(x, t)) \approx \mathbb{E}_v \left[ v^T \nabla_x \mathbf{f}(x, t) v \right]
    $$

    Args:
      key: JAX random key
      t: Time parameter
      x: Input data
      state: Model state

    Returns:
      Tuple of (score, divergence)
    """
    # random vector
    eps = jax.random.randint(key, x.shape, 0, 2).astype(float) * 2 - 1.0

    def sdlogqdx(_x):
        # NN = \sigma_t\nabla_x\log q_t(x)
        return state.apply_fn(state.params, t, _x)

    sdlogdx_val, jvp_val = jax.jvp(sdlogqdx, (x,), (eps,))
    return sdlogdx_val, (jvp_val * eps).sum(1, keepdims=True)


def generate_samples(key, state):
    """
    Generate samples from a trained diffusion model using Euler-Maruyama.

    Args:
      key: JAX random key
      state: Trained model state

    Returns:
      Generated samples trajectory of shape (bs, n+1, ndim)
    """
    dt = 1e-2
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
    for i in trange(n):
        key, ikey = random.split(key, num=2)
        # dx_t = f_t(x_t) dt + g_t dW_t
        # g_t = sqrt( 2 * sigma_t^2 d/dt log(sigma_t / alpha_t) )
        dx = -dt * vector_field_SDE(state, t, x_gen[:, i, :]) + jnp.sqrt(
            2 * jnp.exp(log_sigma(t)) * beta(t) * dt
        ) * random.normal(ikey, shape=(bs, 2))
        x_gen = x_gen.at[:, i + 1, :].set(x_gen[:, i, :] + dx)
        t += -dt
    return x_gen


# Model up - Generate samples from the "up" model

key, ikey = random.split(key)
x_gen = generate_samples(ikey, state_up)
x_gen_up = jnp.copy(x_gen[:, -1, :])

# Visualize the reverse diffusion process for "up" model
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data", alpha=0.3)
    plt.scatter(
        x_gen[:, int(x_gen.shape[1] * (t_axis[i])), 0],
        x_gen[:, int(x_gen.shape[1] * (t_axis[i])), 1],
        label="gen_data",
        color="green",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-up.png")
plt.savefig(fname)
print(f"Saved {fname}")

# Model down - Generate samples from the "down" model

key, ikey = random.split(key)
x_gen = generate_samples(ikey, state_down)
x_gen_down = jnp.copy(x_gen[:, -1, :])

# Visualize the reverse diffusion process for "down" model
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data", alpha=0.3)
    plt.scatter(
        x_gen[:, int(x_gen.shape[1] * (t_axis[i])), 0],
        x_gen[:, int(x_gen.shape[1] * (t_axis[i])), 1],
        label="gen_data",
        color="green",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-down.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Sampling from the isosurface (equal density of both models) (AND)


@jax.jit
def get_dll(t, x, sdlogdx_val, divlog_val, dxdt):
    r"""
    Compute the change in log-likelihood.
    
    $$
    d \log q_{1 - \tau}(x_\tau) =
    \left\langle dx_\tau, \nabla \log q_{1 - \tau}(x_\tau) \right\rangle
    + \left\langle \nabla, f_{1 - \tau}(x_\tau) \right\rangle 
    +
    \left\langle f_{1 - \tau}(x_\tau) - \frac{g_{1 - \tau}^2}{2} \nabla \log q_{1 - \tau}(x_\tau), \nabla \log q_{1 - \tau}(x_\tau) \right\rangle d\tau
    $$
    t = 1 - \tau

    Args:
      t: Time parameter
      x: Input data
      sdlogdx_val: Score value
      divlog_val: Divergence value
      dxdt: Vector field

    Returns:
      Log-likelihood derivative
    """
    v = dlog_alphadt(t) * x - beta(t) * sdlogdx_val
    dlldt = -dlog_alphadt(t) * ndim + beta(t) * divlog_val
    dlldt += -((sdlogdx_val / jnp.exp(log_sigma(t))) * (v - dxdt)).sum(1, keepdims=True)
    return dlldt


@jax.jit
def get_kappa(t, divlogs, sdlogdxs):
    r"""
    Compute the relative weights kappa of different models for isosurface sampling,
    which results in equal (change of) density for every model

    Proposition 6
    For the SDE
    $$
    dx_{\tau} = \sum_{j=1}^{M} \kappa_j u_{\tau}^{j}(x_{\tau}) d\tau + g_{1-\tau} d\bar{W}_{\tau},
    $$
    where $\kappa$ are the weights of different models and $\sum_{j} \kappa_j = 1$,
    one can find $\kappa$ that satisfies
    $$
    d \log q_{1-\tau}^{i}(x_{\tau}) = d \log q_{1-\tau}^{j}(x_{\tau}), \quad \forall \ i, j \in [M],
    $$
    by solving a system of $M + 1$ linear equations w.r.t. $\kappa$.

    Args:
      t: Time parameter
      divlogs: Tuple of divergence values from both models
      sdlogdxs: Tuple of score values from both models

    Returns:
      relative weights kappa of different models
    """
    divlog_1, divlog_2 = divlogs
    sdlogdx_1, sdlogdx_2 = sdlogdxs
    kappa = jnp.exp(log_sigma(t)) * (divlog_1 - divlog_2) + (
        sdlogdx_1 * (sdlogdx_1 - sdlogdx_2)
    ).sum(1, keepdims=True)
    kappa /= ((sdlogdx_1 - sdlogdx_2) ** 2).sum(1, keepdims=True)
    return kappa


bs = 512

# Isosurface sampling implementation
dt = 1e-3
t = 1.0
n = int(t / dt)
t = t * jnp.ones((bs, 1))
key, ikey = random.split(key, num=2)
x_gen = jnp.zeros((bs, n + 1, ndim))
x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
ll_1 = np.zeros((bs, n + 1))
ll_2 = np.zeros((bs, n + 1))
for i in trange(n):
    x_t = x_gen[:, i, :]
    key, ikey = random.split(key, num=2)
    sdlogdx_1, divdlog_1 = score_and_divergence(ikey, t, x_t, state_up)
    sdlogdx_2, divdlog_2 = score_and_divergence(ikey, t, x_t, state_down)
    # solve linear equations for kappa, proposition 6
    kappa = get_kappa(t, (divdlog_1, divdlog_2), (sdlogdx_1, sdlogdx_2))
    dxdt = dlog_alphadt(t) * x_t - beta(t) * (
        sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2)
    )
    x_gen = x_gen.at[:, i + 1, :].set(x_t - dt * dxdt)
    # update log-likelihoods
    ll_1[:, i + 1] = (
        ll_1[:, i] - dt * get_dll(t, x_t, sdlogdx_1, divdlog_1, dxdt).squeeze()
    )
    ll_2[:, i + 1] = (
        ll_2[:, i] - dt * get_dll(t, x_t, sdlogdx_2, divdlog_2, dxdt).squeeze()
    )
    t += -dt


# Plot log-likelihood differences
plt.plot((ll_1 - ll_2)[:20, :].T)
plt.grid()
fname = os.path.join(plotfolder, "toy-add-ll-difference.png")
plt.savefig(fname)
print(f"Saved {fname}")

# Visualize isosurface sampling results
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_up", alpha=0.3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_down", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-add.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Simple averaging - 50/50 mixture of both models

bs = 512

# Simple 50/50 averaging implementation
dt = 1e-3
t = 1.0
n = int(t / dt)
t = t * jnp.ones((bs, 1))
key, ikey = random.split(key, num=2)
x_gen = jnp.zeros((bs, n + 1, ndim))
x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
ll_1 = np.zeros((bs, n + 1))
ll_2 = np.zeros((bs, n + 1))
for i in trange(n):
    x_t = x_gen[:, i, :]
    key, ikey = random.split(key, num=2)
    sdlogdx_1, divdlog_1 = score_and_divergence(ikey, t, x_t, state_up)
    sdlogdx_2, divdlog_2 = score_and_divergence(ikey, t, x_t, state_down)
    kappa = 0.5  # Fixed 50/50 mixture
    dxdt = dlog_alphadt(t) * x_t - beta(t) * (
        sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2)
    )
    x_gen = x_gen.at[:, i + 1, :].set(x_t - dt * dxdt)
    ll_1[:, i + 1] = (
        ll_1[:, i] - dt * get_dll(t, x_t, sdlogdx_1, divdlog_1, dxdt).squeeze()
    )
    ll_2[:, i + 1] = (
        ll_2[:, i] - dt * get_dll(t, x_t, sdlogdx_2, divdlog_2, dxdt).squeeze()
    )
    t += -dt

x_gen_avg = jnp.copy(x_gen)


# Plot log-likelihood differences and ratios
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot((ll_1 - ll_2)[:20, :].T)
plt.xlabel("Time")
plt.ylabel("Log-likelihood difference")
plt.grid()
plt.subplot(122)
plt.plot((jnp.exp(ll_1) / (jnp.exp(ll_1) + jnp.exp(ll_2)))[:20, :].T)
plt.xlabel("Time")
plt.ylabel("Log-likelihood ratio")
plt.grid()
fname = os.path.join(plotfolder, "toy-avg-ll-difference.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Visualize simple averaging results
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_up", alpha=0.3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_down", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-avg.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Averaging proportionally to the density (OR)

bs = 512

# Density-proportional averaging implementation
dt = 1e-3
t = 1.0
n = int(t / dt)
t = t * jnp.ones((bs, 1))
key, ikey = random.split(key, num=2)
x_gen = jnp.zeros((bs, n + 1, ndim))
x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
ll_1 = np.zeros((bs, n + 1))
ll_2 = np.zeros((bs, n + 1))
for i in trange(n):
    x_t = x_gen[:, i, :]
    key, ikey = random.split(key, num=2)
    sdlogdx_1, divdlog_1 = score_and_divergence(ikey, t, x_t, state_up)
    sdlogdx_2, divdlog_2 = score_and_divergence(ikey, t, x_t, state_down)
    # Compute kappa based on density ratio
    max_ll = jnp.maximum(ll_1[:, i], ll_2[:, i])
    kappa = jnp.exp(ll_1[:, i] - max_ll) / (
        jnp.exp(ll_1[:, i] - max_ll) + jnp.exp(ll_2[:, i] - max_ll)
    )
    kappa = kappa[:, None]
    dxdt = dlog_alphadt(t) * x_t - beta(t) * (
        sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2)
    )
    x_gen = x_gen.at[:, i + 1, :].set(x_t - dt * dxdt)
    ll_1[:, i + 1] = (
        ll_1[:, i] - dt * get_dll(t, x_t, sdlogdx_1, divdlog_1, dxdt).squeeze()
    )
    ll_2[:, i + 1] = (
        ll_2[:, i] - dt * get_dll(t, x_t, sdlogdx_2, divdlog_2, dxdt).squeeze()
    )
    t += -dt


# Plot log-likelihood differences and ratios
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot((ll_1 - ll_2)[:20, :].T)
plt.title("log-likelihood difference (20 samples)")
plt.xlabel("time")
plt.ylabel("log-likelihood")
plt.grid()
plt.subplot(122)
plt.plot((jnp.exp(ll_1) / (jnp.exp(ll_1) + jnp.exp(ll_2)))[:20, :].T)
plt.title("log-likelihood ratio (20 samples)")
plt.xlabel("time")
plt.ylabel("log-likelihood")
plt.grid()
fname = os.path.join(plotfolder, "toy-or-ll-difference.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Visualize density-proportional averaging results
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_up", alpha=0.3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_down", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-or.png")
plt.savefig(fname)
print(f"Saved {fname}")


# Separate points by which model assigns higher likelihood
up_mask = ll_1[:, -1] > ll_2[:, -1]
down_mask = ll_1[:, -1] <= ll_2[:, -1]

plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    plt.scatter(
        x_gen[up_mask, int(n * (t_axis[i])), 0],
        x_gen[up_mask, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.scatter(
        x_gen[down_mask, int(n * (t_axis[i])), 0],
        x_gen[down_mask, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(plotfolder, "toy-or-separate.png")
plt.savefig(fname)
print(f"Saved {fname}")

# Stochastic Superposition
@jax.jit
def get_sscore(state, t, x):
    r"""
    Get the score from a model state.
    
    NN = \sigma_t \nabla_x \log q_t(x)
    
    Args:
      state: Model state
      t: Time parameter
      x: Input data

    Returns:
      Score value
    """
    return state.apply_fn(state.params, t, x)


@jax.jit
def get_stoch_dll(t, dt, x, dx, sscore):
    """
    Compute stochastic log-likelihood derivative.

    Args:
      t: Time parameter
      dt: Time step
      x: Input data
      dx: Change in x
      sscore: Score value

    Returns:
      Stochastic log-likelihood derivative
    """
    output = ndim * dt * dlog_alphadt(t) - dt * beta(t) * (sscore**2) / jnp.exp(
        log_sigma(t)
    )
    output += (dx + dt * dlog_alphadt(t) * x) * sscore / jnp.exp(log_sigma(t))
    return output.sum(1)


# Stochastic superposition implementation with softmax weighting (OR)
def generate_or(key, state_up, state_down, x_t, bs = 512):
    dt = 1e-3
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
    ll_1 = np.zeros((bs, n + 1))
    ll_2 = np.zeros((bs, n + 1))
    for i in trange(n):
        x_t = x_gen[:, i, :]
        key, ikey = random.split(key, num=2)
        sdlogdx_1 = get_sscore(state_up, t, x_t)
        sdlogdx_2 = get_sscore(state_down, t, x_t)
        # Use softmax to compute kappa
        kappa = jax.nn.softmax(jnp.stack([ll_1[:, i], ll_2[:, i]]), axis=0)[0]
        kappa = kappa[:, None]
        dx = -dt * (
            dlog_alphadt(t) * x_t
            - 2 * beta(t) * (sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2))
        )
        dx += jnp.sqrt(2 * jnp.exp(log_sigma(t)) * beta(t) * dt) * random.normal(
            ikey, shape=(bs, 2)
        )
        x_gen = x_gen.at[:, i + 1, :].set(x_t + dx)
        ll_1[:, i + 1] = ll_1[:, i] + get_stoch_dll(t, dt, x_t, dx, sdlogdx_1).squeeze()
        ll_2[:, i + 1] = ll_2[:, i] + get_stoch_dll(t, dt, x_t, dx, sdlogdx_2).squeeze()
        t += -dt
    return x_gen

x_gen_or = generate_or(key, state_up, state_down, x_t)


plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot((ll_1 - ll_2)[:20, :].T)
plt.grid()
plt.subplot(122)
plt.plot((jnp.exp(ll_1) / (jnp.exp(ll_1) + jnp.exp(ll_2)))[:20, :].T)
plt.grid()


plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_up", alpha=0.3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_down", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)


# @jax.jit
# def get_stoch_dll(t, dt, x, dx, sscore):
#     output = ndim * dt * dlog_alphadt(t) - dt * beta(t) * (sscore**2) / jnp.exp(
#         log_sigma(t)
#     )
#     output += (dx + dt * dlog_alphadt(t) * x) * sscore / jnp.exp(log_sigma(t))
#     return output.sum(1)


@jax.jit
def select_kappa(ikey, t, dt, x, sdlogdx_1, sdlogdx_2):
    noise = jnp.sqrt(2 * jnp.exp(log_sigma(t)) * beta(t) * dt) * random.normal(
        ikey, shape=(bs, 2)
    )
    dx_ind = -dt * (dlog_alphadt(t) * x - 2 * beta(t) * sdlogdx_2) + noise
    kappa = (
        -dt
        * beta(t)
        * (sdlogdx_1 - sdlogdx_2)
        * (sdlogdx_1 + sdlogdx_2)
        / jnp.exp(log_sigma(t))
    )
    kappa += (
        (dx_ind + dt * dlog_alphadt(t) * x)
        * (sdlogdx_1 - sdlogdx_2)
        / jnp.exp(log_sigma(t))
    )
    kappa = -kappa.sum(1) / (
        dt * 2 * beta(t) * (sdlogdx_1 - sdlogdx_2) ** 2 / jnp.exp(log_sigma(t))
    ).sum(1)
    return kappa

def generate_and(key, state_up, state_down):
    eta = 0.9
    dt = 1e-3
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
    ll_0 = -0.5 * (x_gen[:, 0, :] ** 2).sum(1) - ndim * jnp.log(2 * jnp.pi)
    ll_1 = ll_0[:, None] * jnp.ones((bs, n + 1))
    ll_2 = ll_0[:, None] * jnp.ones((bs, n + 1))
    for i in trange(n):
        x_t = x_gen[:, i, :]
        key, ikey = random.split(key, num=2)
        sdlogdx_1 = get_sscore(state_up, t, x_t)
        sdlogdx_2 = get_sscore(state_down, t, x_t)
        kappa = select_kappa(ikey, t, dt, x_t, sdlogdx_1, sdlogdx_2)
        kappa = kappa[:, None]
        dx = -dt * (
            dlog_alphadt(t) * x_t
            - 2 * beta(t) * (sdlogdx_2 + kappa * (sdlogdx_1 - sdlogdx_2))
        )
        dx += jnp.sqrt(2 * jnp.exp(log_sigma(t)) * beta(t) * dt) * random.normal(
            ikey, shape=(bs, 2)
        )
        x_gen = x_gen.at[:, i + 1, :].set(x_t + dx)
        ll_1 = ll_1.at[:, i + 1].set(
            ll_1[:, i] + get_stoch_dll(t, dt, x_t, dx, sdlogdx_1).squeeze()
        )
        ll_2 = ll_2.at[:, i + 1].set(
            ll_2[:, i] + get_stoch_dll(t, dt, x_t, dx, sdlogdx_2).squeeze()
        )
        t += -dt

    return x_gen, ll_1, ll_2

x_gen_and, ll_1, ll_2 = generate_and(key, state_up, state_down)

plt.close()
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot((ll_1 - ll_2)[:20, :].T)
plt.title("log-likelihood difference (20 samples)")
plt.xlabel("time")
plt.ylabel("log-likelihood")
plt.grid()
plt.subplot(122)
plt.plot((jnp.exp(ll_1) / (jnp.exp(ll_1) + jnp.exp(ll_2)))[:20, :].T)
plt.title("log-likelihood ratio (20 samples)")
plt.xlabel("time")
plt.ylabel("log-likelihood")
plt.grid()
fname = "fig1_toy_example/toy-and.png"
plt.savefig(fname)
print(f"Saved {fname}")


plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=True), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_up", alpha=0.3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs // 2, up=False), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data_down", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)


C1 = "#1D9D79"
C2 = "#756FB3"
C3 = "#D96002"
alpha = 0.6
alpha_3 = 0.7


def generate_plot(C1, C2, C3):
    # Set random seed for reproducibility
    np.random.seed(42)
    # Create a figure with 1 row and 4 columns of subplots
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    # First subplot
    axs[0].scatter(x_gen_up[:, 0], x_gen_up[:, 1], c=C1, alpha=alpha)
    axs[0].scatter(x_gen_down[:, 0], x_gen_down[:, 1], c=C2, alpha=alpha)
    # Second subplot
    # simple average
    axs[1].scatter(x_gen_up[:, 0], x_gen_up[:, 1], c=C1, alpha=alpha)
    axs[1].scatter(x_gen_down[:, 0], x_gen_down[:, 1], c=C2, alpha=alpha)
    axs[1].scatter(
        x_gen_avg[:, int(n * (t_axis[5])), 0],
        x_gen_avg[:, int(n * (t_axis[5])), 1],
        c=C3,
        alpha=alpha_3,
    )
    # Third subplot 
    # (OR)
    axs[2].scatter(x_gen_up[:, 0], x_gen_up[:, 1], c=C1, alpha=alpha)
    axs[2].scatter(x_gen_down[:, 0], x_gen_down[:, 1], c=C2, alpha=alpha)
    axs[2].scatter(
        x_gen_or[:, int(n * (t_axis[5])), 0],
        x_gen_or[:, int(n * (t_axis[5])), 1],
        c=C3,
        alpha=alpha_3,
    )
    # Fourth subplot
    # equal density (AND)
    axs[3].scatter(x_gen_up[:, 0], x_gen_up[:, 1], c=C1, alpha=alpha)
    axs[3].scatter(x_gen_down[:, 0], x_gen_down[:, 1], c=C2, alpha=alpha)
    axs[3].scatter(
        x_gen_and[:, int(n * (t_axis[5])), 0],
        x_gen_and[:, int(n * (t_axis[5])), 1],
        c=C3,
        alpha=alpha_3,
    )
    # Set limits and titles for each subplot
    ms = 1.6
    for ax in axs:
        ax.set_xlim(-ms * 2, ms * 2)
        ax.set_ylim(-ms * 2, ms * 2)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_yticklabels([])  # Remove y-axis labels

    dummy_l1 = axs[4].scatter([], [], c=C1, label="Train Data A", s=90.0, alpha=1)
    dummy_l2 = axs[4].scatter([], [], c=C2, label="Train Data B", s=90.0, alpha=1)
    dummy_l3 = axs[4].scatter([], [], c=C3, label="Generated samples", s=90.0, alpha=1)

    axs[4].spines["top"].set_visible(False)
    axs[4].spines["bottom"].set_visible(False)
    axs[4].spines["left"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    # Add a single legend outside the plots on the right
    fig.legend(
        handles=[dummy_l1, dummy_l2, dummy_l3],
        loc="center left",
        bbox_to_anchor=(0.767, 0.5),
        ncol=3,
        prop={"family": "monospace", "size": 16},
    )

    # Show the plot
    # plt.tight_layout()
    os.makedirs("fig1_toy_example", exist_ok=True)
    for i, ax in enumerate(axs):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if i == 0:
            plt.savefig(
                "fig1_toy_example/toy-data.pdf",
                dpi=300,
                bbox_inches=extent.expanded(1.1, 1.1),
            )
        elif i == 1:
            plt.savefig(
                "fig1_toy_example/toy-average.pdf",
                dpi=300,
                bbox_inches=extent.expanded(1.1, 1.1),
            )
        elif i == 2:
            plt.savefig(
                "fig1_toy_example/toy-mixture.pdf",
                dpi=300,
                bbox_inches=extent.expanded(1.1, 1.1),
            )
        elif i == 3:
            plt.savefig(
                "fig1_toy_example/toy-eq-density.pdf",
                dpi=300,
                bbox_inches=extent.expanded(1.1, 1.1),
            )
        elif i == 4:
            print(extent)
            extent.x1 += 5.5
            extent.y0 = 2.2
            extent.y1 = 2.8
            plt.savefig("fig1_toy_example/toy-legend.pdf", dpi=300, bbox_inches=extent)
    plt.show()


generate_plot(C1, C2, C3)
