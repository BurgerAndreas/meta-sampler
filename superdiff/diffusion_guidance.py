import jax
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap
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

plotfolder = os.path.join(os.path.dirname(__file__), "figs", "diffusion")
os.makedirs(plotfolder, exist_ok=True)
os.makedirs(os.path.join(plotfolder, "classifier"), exist_ok=True)


# # Diffusion models
r"""
## Intro
For every object in our data distribution $\mu^i \sim p_{\text{data}}(\mu)$ we define the noising process as
$$q_t^i(x) = \mathcal{N}(x|\alpha_t\mu^i,\sigma_t^2)\,.$$
The important part about defining this process is that we know the vector field satisfying the corresponding continuity equation
$$v_t^i(x) = (x-\alpha_t\mu^i)\frac{\partial}{\partial t}\log\sigma_t + \frac{\partial\alpha_t}{\partial t}\mu^i\,.$$
For the entire dataset we can write down the following marginal density
$$q_t(x) = \int \mathcal{N}(x|\alpha_t\mu,\sigma_t^2)p_{\text{data}}(\mu)d\mu\,,$$
and we can write down the loss for the vector field as
$$\text{Loss} = \frac{1}{N}\sum_{i=1}^N\int dt\;\int dx\;\Vert v_t(x;\theta) - v_t^i(x)\Vert^2\,,$$
where $\theta$ are the parameters of our model.

After training, we can simulate the process (generate data) via the following ODE
$$\frac{dx}{dt} = v_t(x)\,,\;\;\; v_t(x) = x\frac{\partial}{\partial t}\log\alpha_t - \beta_t\nabla_x\log q_t(x)\,,\;\;\beta_t = \sigma_t^2\frac{\partial}{\partial t}\log \frac{\sigma_t}{\alpha_t}\,,$$
or SDE
$$dx = (v_t(x) + \xi_t\nabla_x\log q_t(x))\cdot dt + \sqrt{2\xi_t}dW_t\,.$$
"""

# Data Generation
r"""
$p_{\text{data}}(\mu)$ is just a mixture of four Gaussians. Then we define the noising process
$$q_t^i(x) = \mathcal{N}(x|\alpha_t\mu^i,\sigma_t^2)\,.$$
See code for $\alpha_t, \sigma_t$.
The plots represent samples from the following marginals for different times $t$
$$q_t(x) = \int \mathcal{N}(x|\alpha_t\mu,\sigma_t^2)p_{\text{data}}(\mu)d\mu\,.$$
"""

# def sample_data(key, bs):
#     keys = random.split(key, 3)
#     x_1 = random.randint(keys[0], minval=0, maxval=2, shape=(bs, 2))
#     x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
#     x_1 += 4e-1 * random.normal(keys[1], shape=(bs, 2))
#     return x_1


def sample_data(key, bs, up=None):
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
    if up == True:
        x_1 = random.randint(
            keys[0], minval=jnp.array([0, 1]), maxval=jnp.array([2, 2]), shape=(bs, 2)
        )
    elif up == False:
        x_1 = random.randint(
            keys[0], minval=jnp.array([0, 0]), maxval=jnp.array([2, 1]), shape=(bs, 2)
        )
    else:
        x_1 = random.randint(keys[0], minval=0, maxval=2, shape=(bs, 2))
    x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
    x_1 += 4e-1 * random.normal(keys[1], shape=(bs, 2))
    return x_1


def log_alpha(t):
    return -0.5 * t * beta_0 - 0.25 * t**2 * (beta_1 - beta_0)


# log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
def log_sigma(t):
    return jnp.log(t)


dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())


# beta_t = s_t d/dt log(s_t/alpha_t)
# beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
def beta(t):
    return 1 + 0.5 * t * beta_0 + 0.5 * t**2 * (beta_1 - beta_0)


def q_t(key, data, t):
    eps = random.normal(key, shape=data.shape)
    x_t = jnp.exp(log_alpha(t)) * data + jnp.exp(log_sigma(t)) * eps
    return eps, x_t


def plot_noising_process(key, bs, t_axis):
    # plot the noising process
    plt.figure(figsize=(23, 5))
    for i in range(len(t_axis)):
        plt.subplot(1, len(t_axis), i + 1)
        key, *ikey = random.split(key, 3)
        _, x_t = q_t(ikey[1], sample_data(ikey[0], bs), t_axis[i])
        plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.3)
        plt.title(f"t={t_axis[i]}")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
    plt.tight_layout()
    fname = os.path.join(plotfolder, "noising_process.png")
    plt.savefig(fname)
    print(f"Saved\n {fname}")


# Define the Model
r"""
For the model of the vector field $v_t(x;\theta)$, we take an MLP.
"""


class MLP(nn.Module):
    num_hid: int
    num_out: int

    @nn.compact
    def __call__(self, t, x):
        h = jnp.hstack([t, x])
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.relu(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_out)(h)
        return h


# Classifier for up/down regions
class Classifier(nn.Module):
    num_hid: int

    @nn.compact
    def __call__(self, x):
        h = x
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.relu(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=1)(h)
        return h


# Loss Function
r"""
In general, the loss looks like this
$$\text{Loss} = \frac{1}{N}\sum_{i=1}^N\int dt\;\int dx\;\Vert v_t(x;\theta) - v_t^i(x)\Vert^2\,.$$
However, we rewrite it a bit in terms of the score and parametrize the following quantity
$$\text{MLP}(t,x;\theta) = \sigma_t\nabla_x\log q_t(x)\,.$$
"""


def sm_loss(state, key, params, bs):
    keys = random.split(
        key,
    )

    def sdlogqdx(_t, _x):
        return state.apply_fn(params, _t, _x)

    data = sample_data(keys[0], bs)
    t = random.uniform(keys[1], [bs, 1])
    eps, x_t = q_t(keys[2], data, t)
    loss = ((eps + sdlogqdx(t, x_t)) ** 2).sum(1)
    print(loss.shape, "final.shape", flush=True)
    return loss.mean()


@partial(jax.jit, static_argnums=1)
def train_step(state, bs, key):
    grad_fn = jax.value_and_grad(sm_loss, argnums=2)
    loss, grads = grad_fn(state, key, state.params, bs)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Train Loop


def train_diffusion(key, state, bs):
    num_iterations = 20_000

    loss_plot = np.zeros(num_iterations)
    key, loop_key = random.split(key)
    for iter in trange(num_iterations, desc="Training diffusion model"):
        state, loss = train_step(state, bs, random.fold_in(loop_key, iter))
        loss_plot[iter] = loss
    return state, loss_plot


def plot_loss(loss_plot):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_plot)
    plt.grid()
    fname = os.path.join(plotfolder, "loss.png")
    plt.savefig(fname)
    print(f"Saved\n {fname}")


# Evaluation of the Trained Model
r"""
For the evaluation we simply sample from $q_{t=1}$ and propogate samples back in time according to the ODE
$$\frac{dx}{dt} = v_t(x)\,,\;\;\; v_t(x) = x\frac{\partial}{\partial t}\log\alpha_t - \beta_t\nabla_x\log q_t(x)\,,\;\;\beta_t = \sigma_t^2\frac{\partial}{\partial t}\log \frac{\sigma_t}{\alpha_t}\,,$$
or SDE
$$dx = (v_t(x) + \xi_t\nabla_x\log q_t(x))\cdot dt + \sqrt{2\xi_t}dW_t\,.$$
"""


@jax.jit
def get_sde_drift(t, x, xi, state):
    """Vector field of the diffusion process

    vector field
    v_t(x) = dlog(alpha)/dt x - beta_t * dlog(q_t(x))/dx
    beta_t = s^2_t d/dt log(s_t/alpha_t)
    
    used to simulate the SDE
    dx = f_t(x) dt + sqrt(2 * eps_t) dW_t
    drift f_t(x) = v_t(x) + eps_t * dlog(q_t(x))/dx
    """
    
    def sdlogqdx(_t, _x):
        # NN = \sigma_t\nabla_x\log q_t(x)
        return state.apply_fn(state.params, _t, _x)

    dxdt = (
        dlog_alphadt(t) * x
        - beta(t) * sdlogqdx(t, x)
        - xi * beta(t) / jnp.exp(log_sigma(t)) * sdlogqdx(t, x)
    )
    return dxdt


def generate_samples(key, vf_state, bs=512):
    dt = 1e-2
    xi = 1.0  # stochasticity parameter
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))

    for i in trange(n, desc="Generating"):
        key, ikey = random.split(key, num=2)
        noise_term = jnp.sqrt(2 * xi * beta(t) * dt) * random.normal(
            ikey, shape=(bs, 2)
        )
        diff_vec_field = -dt * get_sde_drift(t, x_gen[:, i, :], xi, vf_state)
        dx = diff_vec_field + noise_term
        x_gen = x_gen.at[:, i + 1, :].set(x_gen[:, i, :] + dx)
        t += -dt
    return x_gen


def plot_samples(x_gen, t_axis, key):
    plt.figure(figsize=(23, 5))
    n = x_gen.shape[1]
    for i in range(len(t_axis)):
        plt.subplot(1, len(t_axis), i + 1)
        key, *ikey = random.split(key, 3)
        t = t_axis[len(t_axis) - 1 - i]
        _, x_t = q_t(ikey[1], sample_data(ikey[0], bs), t)
        plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data", alpha=0.3)
        plt.scatter(
            x_gen[:, int(n * (t_axis[i])), 0],
            x_gen[:, int(n * (t_axis[i])), 1],
            label="gen_data",
        )
        plt.title(f"t={t:.2f}")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)

    plt.tight_layout()
    fname = os.path.join(plotfolder, "gen_data.png")
    plt.savefig(fname)
    print(f"Saved\n {fname}")


# Train a classifier to separate up and down regions
def train_classifier(key):
    # Initialize classifier
    classifier = Classifier(num_hid=128)
    key, cls_init_key = random.split(key)

    # Initialize training state
    cls_optimizer = optax.adam(learning_rate=1e-3)
    cls_params = classifier.init(cls_init_key, jnp.zeros((1, ndim)))
    cls_state = train_state.TrainState.create(
        apply_fn=classifier.apply,
        params=cls_params,
        tx=cls_optimizer,
    )

    # Define loss function (binary cross entropy)
    @jax.jit
    def cls_loss_fn(params, x, y):
        logits = cls_state.apply_fn(params, x)
        loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
        return loss

    # Define training step
    @jax.jit
    def cls_train_step(state, x, y):
        grad_fn = jax.value_and_grad(cls_loss_fn)
        loss, grads = grad_fn(state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Training loop
    cls_bs = 256
    num_cls_iterations = 1000
    losses = []

    for iter in trange(num_cls_iterations, desc="Training classifier"):
        key, data_key = random.split(key)

        # Generate balanced dataset with up and down samples
        key_up, key_down = random.split(data_key)
        x_up = sample_data(key_up, cls_bs // 2, up=True)
        x_down = sample_data(key_down, cls_bs // 2, up=False)
        x_batch = jnp.concatenate([x_up, x_down], axis=0)
        y_batch = jnp.concatenate(
            [jnp.ones((cls_bs // 2, 1)), jnp.zeros((cls_bs // 2, 1))], axis=0
        )

        # Update classifier
        cls_state, loss = cls_train_step(cls_state, x_batch, y_batch)
        losses.append(loss)

    # remove first 10 losses
    losses = losses[10:]

    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.semilogy(losses)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Classifier Training Loss")
    fname = os.path.join(plotfolder, "classifier", "loss.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

    # Plot decision boundary
    key, vis_key = random.split(key)
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Get classifier predictions on grid
    logits = jax.vmap(lambda x: cls_state.apply_fn(cls_state.params, x))(grid_points)
    probs = jax.nn.sigmoid(logits).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, cmap="RdBu", alpha=0.7)
    plt.colorbar(label="P(up)")

    # Plot real data
    key_vis_up, key_vis_down = random.split(vis_key)
    vis_samples = 200
    x_vis_up = sample_data(key_vis_up, vis_samples, up=True)
    x_vis_down = sample_data(key_vis_down, vis_samples, up=False)
    plt.scatter(x_vis_up[:, 0], x_vis_up[:, 1], color="red", alpha=0.5, label="Up")
    plt.scatter(
        x_vis_down[:, 0], x_vis_down[:, 1], color="blue", alpha=0.5, label="Down"
    )

    plt.legend()
    plt.grid()
    plt.title("Classifier Decision Boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    fname = os.path.join(plotfolder, "classifier", "decision_boundary.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

    return cls_state


def generate_samples_from_vectorfield(
    key,
    vf_fn,
    bs=512,
):
    """Generate samples using classifier guidance for a specific class

    Args:
        key: JAX random key
        vf_fn: Vector field function
        bs: Batch size
    """
    dt = 1e-2
    t = 1.0
    xi = 1.0  # stochasticity parameter
    n = int(t / dt)
    t_batch = t * jnp.ones((bs, 1))
    key, ikey = random.split(key)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
    # denoise / integrate the vector field
    for i in trange(n, desc=f"Generating samples"):
        key, ikey = random.split(key)
        # Standard diffusion vector field
        diff_vec_field = -dt * vf_fn(t=t_batch, x=x_gen[:, i, :], xi=xi)
        # Combined update with noise
        noise_term = jnp.sqrt(2 * xi * beta(t_batch) * dt) * random.normal(
            ikey, shape=(bs, 2)
        )
        dx = diff_vec_field + noise_term
        # Update sample
        x_gen = x_gen.at[:, i + 1, :].set(x_gen[:, i, :] + dx)
        t_batch += -dt
    return x_gen


def classifier_vectorfield(t, x, xi, cls_state, vf_state, target_class, guidance_scale):
    # Define classifier gradient function
    def classifier_grad(x):
        return jax.grad(
            lambda x_input: cls_state.apply_fn(
                cls_state.params, x_input
            ).sum()
        )(x)

    direction = -1.0 if target_class == "up" else 1.0
    cls_grad = jax.vmap(classifier_grad)(x)
    guidance_vec = direction * guidance_scale * t * cls_grad
    return guidance_vec + get_sde_drift(t=t, x=x, xi=xi, state=vf_state)

def plot_samples_with_guidance(x_gen_unguided, x_gen_up, x_gen_down):
    # Plot samples with guidance
    nrows = 3
    plt.figure(figsize=(6, nrows * 6))  # width, height

    # Plot samples without guidance (original generation)
    plt.subplot(nrows, 1, 1)
    plt.scatter(x_gen_unguided[:, -1, 0], x_gen_unguided[:, -1, 1], alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title("Samples with zero guidance")

    # Plot samples with guidance
    plt.subplot(nrows, 1, 2)
    plt.scatter(x_gen_up[:, -1, 0], x_gen_up[:, -1, 1], alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title(f"Samples with up guidance (scale={guidance_scale})")

    # Plot samples with guidance
    plt.subplot(nrows, 1, 3)
    plt.scatter(x_gen_down[:, -1, 0], x_gen_down[:, -1, 1], alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.title(f"Samples with down guidance (scale={guidance_scale})")
    plt.tight_layout()
    fname = os.path.join(plotfolder, "classifier", "guided_vs_unguided.png")
    plt.savefig(fname)
    print(f"Saved {fname}")


# Vector field computation with Jacobian-vector product
# v_t(x) = dlog(alpha)/dt x - s^2_t d/dt log(s_t/alpha_t) dlog q_t(x)/dx
@jax.jit
def vector_field_jvp(key, t, x, state):
    """
    Compute the vector field and its divergence using JVP.

    Args:
      key: JAX random key
      t: Time parameter
      x: Input data
      state: Model state

    Returns:
      Tuple of (score, divergence)
    """
    eps = jax.random.randint(key, x.shape, 0, 2).astype(float) * 2 - 1.0

    def sdlogqdx(_x):
        return state.apply_fn(vf_state.params, t, _x)

    sdlogdx_val, jvp_val = jax.jvp(sdlogqdx, (x,), (eps,))
    return sdlogdx_val, (jvp_val * eps).sum(1, keepdims=True)


# Sampling from the isosurface (equal density of both models) (AND)

@jax.jit
def get_dll(t, x, sdlogdx_val, divlog_val, dxdt):
    r"""
    Compute the log-likelihood derivative.

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


def add_isosurface(vf_jvp_fn1, vf_jvp_fn2,key, bs=512):
    # Isosurface sampling implementation
    dt = 1e-3
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, x_t.shape[1]))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, x_t.shape[1])))
    ll_1 = np.zeros((bs, n + 1))
    ll_2 = np.zeros((bs, n + 1))
    for i in trange(n):
        x_t = x_gen[:, i, :]
        key, ikey = random.split(key, num=2)
        sdlogdx_1, divdlog_1 = vf_jvp_fn1(ikey, t, x_t)
        sdlogdx_2, divdlog_2 = vf_jvp_fn2(ikey, t, x_t)
        # solve linear equations for kappa, proposition 6
        kappa = get_kappa(t, (divdlog_1, divdlog_2), (sdlogdx_1, sdlogdx_2))
        # f_t(x_t) = d/dt log alpha_t x_t
        # g_t = sqrt( 2 * sigma_t^2 d/dt log(sigma_t / alpha_t) )
        # dx_t = f_t(x_t) dt + g_t dW_t
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

# @jax.jit
def get_guided_sscore(t, x, target_class, guidance_scale):
    direction = -1.0 if target_class == "up" else 1.0
    cls_grad = jax.vmap(jax.grad(
        lambda x_input: cls_state.apply_fn(
            cls_state.params, x_input
        ).sum()
    ))(x)
    guidance_vec = direction * guidance_scale * t * cls_grad
    return guidance_vec + get_sscore(vf_state, t, x)

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

def generate_and(key, sscore_fn1, sscore_fn2):
    eta = 0.9
    dt = 1e-2 # 1e-3
    t = 1.0
    n = int(t / dt)
    t = t * jnp.ones((bs, 1))
    key, ikey = random.split(key, num=2)
    x_gen = jnp.zeros((bs, n + 1, ndim))
    x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, ndim)))
    ll_0 = -0.5 * (x_gen[:, 0, :] ** 2).sum(1) - ndim * jnp.log(2 * jnp.pi)
    ll_1 = ll_0[:, None] * jnp.ones((bs, n + 1))
    ll_2 = ll_0[:, None] * jnp.ones((bs, n + 1))
    for i in trange(n, desc="Stochastic Superposition"):
        x_t = x_gen[:, i, :]
        key, ikey = random.split(key, num=2)
        sdlogdx_1 = sscore_fn1(t, x_t)
        sdlogdx_2 = sscore_fn2(t, x_t)
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

def plot_and(x_gen, ll_1, ll_2, data1, data2, key):
    plt.figure(figsize=(23, 5))
    t_axis = np.linspace(0.0, 1.0, 6)
    n = data1.shape[1]
    for i in range(len(t_axis)):
        plt.subplot(1, len(t_axis), i + 1)
        key, *ikey = random.split(key, 3)
        t = t_axis[len(t_axis) - 1 - i]
        plt.scatter(data1[:, -1, 0], data1[:, -1, 1], label="1", alpha=0.3, color=C1)
        plt.scatter(data2[:, -1, 0], data2[:, -1, 1], label="2", alpha=0.3, color=C2)
        plt.scatter(
            x_gen[:, int(n * (t_axis[i])), 0],
            x_gen[:, int(n * (t_axis[i])), 1],
            label="gen_data",
            color=C3,
        )
        plt.title(f"t={t}")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.tight_layout()
    fname = os.path.join(plotfolder, "toy-and.png")
    plt.savefig(fname)
    print(f"Saved {fname}")


if __name__ == "__main__":
    
    C1 = "#1D9D79"
    C2 = "#756FB3"
    C3 = "#D96002"

    ndim = 2
    t_0, t_1 = 0.0, 1.0
    beta_0 = 0.1
    beta_1 = 20.0

    seed = 0
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    bs = 512
    t_axis = np.linspace(0.0, 1.0, 6)

    plot_noising_process(key, bs, t_axis)

    model = MLP(num_hid=512, num_out=ndim)
    print(model)

    _, sample_x_t = q_t(key, sample_data(key, bs), 1.0)
    key, init_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    vf_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, np.ones([bs, 1]), sample_x_t),
        tx=optimizer,
    )

    key, loc_key = random.split(key)
    vf_state, loss = train_step(vf_state, bs, loc_key)

    vf_state, loss = train_diffusion(key, vf_state, bs)
    plot_loss(loss)

    key, ikey = random.split(key)
    x_gen = generate_samples(ikey, vf_state)
    plot_samples(x_gen, t_axis, key)
    
    ###################################################################
    # Classifier guidance
    ###################################################################

    # Train classifier
    # TODO: 4 classes instead of 2
    key, cls_key = random.split(key)
    cls_state = train_classifier(cls_key)

    key, gen_key_no_guide, gen_key_guide = random.split(key, 3)

    x_gen_unguided = generate_samples_from_vectorfield(
        gen_key_guide,
        vf_fn=lambda t, x, xi: classifier_vectorfield(t, x, xi, cls_state, vf_state, "up", 0.0),
        bs=bs,
    )

    # Generate samples with and without guidance
    key, gen_key_no_guide, gen_key_guide = random.split(key, 3)

    x_gen_unguided = generate_samples_from_vectorfield(
        gen_key_guide,
        vf_fn=lambda t, x, xi: classifier_vectorfield(t, x, xi, cls_state, vf_state, "up", 0.0),
        bs=bs,
    )
    
    # Generate samples with classifier guidance
    guidance_scale = 2.0  # Adjust strength as needed

    # Generate "up" class samples with strong guidance
    x_gen_up = generate_samples_from_vectorfield(
        gen_key_guide,
        vf_fn=lambda t, x, xi: classifier_vectorfield(
            t, x, xi, cls_state, vf_state, "up", guidance_scale
        ),
        bs=bs,
    )

    # Generate "down" class samples with strong guidance
    x_gen_down = generate_samples_from_vectorfield(
        gen_key_guide,
        vf_fn=lambda t, x, xi: classifier_vectorfield(
            t, x, xi, cls_state, vf_state, "down", guidance_scale
        ),
        bs=bs,
    )
    plot_samples_with_guidance(x_gen_unguided, x_gen_up, x_gen_down)
    
    ###################################################################
    # Superposition
    ###################################################################
    
    # TODO: vector_field_jvp with guidance
    # vector_field_jvp = partial(vector_field_jvp, state=vf_state)
    # add_isosurface(
    #     vector_field_jvp, 
    #     vector_field_jvp, 
    #     key
    # )
    
    # TODO: stochastic superposition
    x_gen_and, ll_1, ll_2 = generate_and(
        key, 
        lambda t, x_t: get_guided_sscore(t, x_t, "up", 2.0),
        lambda t, x_t: get_guided_sscore(t, x_t, "down", 2.0)
    )
    plot_and(x_gen_and, ll_1, ll_2, x_gen_up, x_gen_down, key)
    