# Transitions from pretrained Boltzmann generators using superdiff

- fix superposition of classifier guidance
- extend classifier guidance to arbitrary number of classes
- replace classifier guidance, e.g. with guidance w.r.t. reference structures
- potentials from https://github.com/necludov/super-diffusion/blob/main/notebooks/superposition_AND.ipynb

### Installation
```bash
wget -O superposition_edu.ipynb https://raw.githubusercontent.com/necludov/super-diffusion/main/notebooks/superposition_edu.ipynb

wget -O diffusion_edu.ipynb https://raw.githubusercontent.com/necludov/super-diffusion/main/notebooks/diffusion_edu.ipynb

wget -O superposition_AND.ipynb https://raw.githubusercontent.com/necludov/super-diffusion/main/notebooks/superposition_AND.ipynb
```


## Superdiff

### Divergence in diffusion models
In the context of diffusion models, particularly score-based generative models (SGMs), the function \( \mathbf{f}(x, t) \) typically represents the score function, i.e., the gradient of the log probability density of the noisy data:

$$
\mathbf{f}(x, t) = \nabla_x \log p_t(x)
$$
During training, this is approximated by a neural network \( s_\theta(x, t) \), so:
$$
\mathbf{f}(x, t) \approx s_\theta(x, t)
$$

Training objectives for these models often involve the divergence of this score function:
$$
\nabla \cdot \mathbf{f}(x, t) = \sum_i \frac{\partial f_i(x, t)}{\partial x_i}
$$

This is the trace of the Jacobian matrix of \( \mathbf{f} \), denoted \( \text{Tr}(\nabla_x \mathbf{f}(x, t)) \). Computing this directly is expensive when \( x \in \mathbb{R}^d \) with large \( d \), so we use the Hutchinson trace estimator:
$$
\text{Tr}(\nabla_x \mathbf{f}(x, t)) \approx \mathbb{E}_v \left[ v^T \nabla_x \mathbf{f}(x, t) v \right]
$$
where \( v \sim \mathcal{N}(0, I) \) or is sampled from a Rademacher distribution (i.e., random \(\pm 1\) entries).
This estimator allows us to efficiently approximate the divergence using automatic differentiation frameworks, since it requires only a Jacobian-vector product. 
JVPs can be done at the cost of one forward pass using forward-mode automatic differentiation, without materializing the full Jacobian, which would be memory-intensive O(mn).

### Algorithm
Input:

* $M$ pre-trained score models $\nabla_x \log q^i_t(x)$,
* the parameters of the schedule $\alpha_t, \sigma_t$,
* stepsize $d\tau > 0$,
* temperature parameter $T$,
* bias parameter $\ell$, and
* initial noise $z \sim \mathcal{N}(0, \mathbf{I})$.

For $\tau = 0, \ldots, 1$ do 
sample noise
$t = 1 - \tau, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})$
compute relative weights of models / vector field
$$
\kappa^i_\tau \leftarrow
\begin{cases}
\text{softmax}\left(T \log q^i_t(x_\tau) + \ell \right) & \text{// for OR according to Prop. 3} \\
\text{solve Linear Equations} & \text{// for AND according to Prop. 6}
\end{cases}
$$

compute vector field
$$
u_\tau(x_\tau) \leftarrow \sum_{i=1}^{M} \kappa^i_\tau \nabla_x \log q^i_t(x_\tau)
$$

compute SDE
$$
dx_\tau = \left( -f_{1-\tau}(x_\tau) + g_{1-\tau}^2 u_\tau(x_\tau) \right) d\tau + g_{1-\tau} dW_\tau
\quad \text{// using Prop. 1}
$$

update
$$
x_{\tau + d\tau} \leftarrow x_\tau + dx_\tau
$$

compute log-likelihood change
$$
d \log q_{1 - \tau}(x_\tau) =
\left\langle dx_\tau, \nabla \log q_{1 - \tau}(x_\tau) \right\rangle
+ \left\langle \nabla, f_{1 - \tau}(x_\tau) \right\rangle 
+
\left\langle f_{1 - \tau}(x_\tau) - \frac{g_{1 - \tau}^2}{2} \nabla \log q_{1 - \tau}(x_\tau), \nabla \log q_{1 - \tau}(x_\tau) \right\rangle d\tau
\quad \text{// using Thm. 1}
$$
Return $x_\tau$


### Theorems

For simulation using the SDE  
$$
dx_\tau = u_\tau(x_\tau) d\tau + g_{1 - \tau} d\mathbf{W}_\tau
$$
Namely, according to Prop. 4, we use the following vector field:
$$
u_\tau(x) = -f_{1 - \tau}(x) + g^2_{1 - \tau} \sum_{i=1}^M \frac{q^i_t(x)}{\sum_j q^j_t(x)} \nabla \log q^i_{1 - \tau}(x)
$$


\textbf{Proposition 1.} \textit{[Reverse-time SDEs/ODE] Marginal densities $q_t(x)$ induced by Eq.~(1) correspond to the densities induced by the following SDE that goes back in time ($\tau = 1 - t$) with the corresponding initial condition}
\begin{equation}
    d\bm{x}_\tau = \left( -\bm{f}_t(\bm{x}_\tau) + \left( \frac{g_t^2}{2} + \xi_\tau \right) \nabla \log q_t(\bm{x}_\tau) \right) d\tau + \sqrt{2\xi_\tau} d\overline{\bm{W}}_\tau, \quad \bm{x}_{\tau=0} \sim q_1(\bm{x}_0), \tag{2}
\end{equation}
\textit{where $\overline{\bm{W}}_\tau$ is the standard Wiener process in time $\tau$, and $\xi_\tau$ is any positive schedule.}


\textbf{Proposition 2.} \textit{[Ornstein--Uhlenbeck SDE] The time-dependent densities in Eq.~(3) correspond to the marginal densities of the following SDE, with the corresponding initial condition}
\begin{equation}
    d\bm{x}_t = \underbrace{\frac{\partial \log \alpha_t}{\partial t} \bm{x}_t}_{\bm{f}_t(\bm{x}_t)} dt + \underbrace{\sqrt{2\sigma_t^2 \frac{\partial}{\partial t} \log \frac{\sigma_t}{\alpha_t}}}_{\bm{g}_t} d\bm{W}_t, \quad \bm{x}_0 \sim q_0(\bm{x}_0). \tag{4}
\end{equation}

dxt = ∂ log αt/∂t xt | {z } ft(xt) dt + sqrt(2 * σt^2 d/dt log σt/alphat ) gt dWt 
x0 ∼ q0(x0)

the simplicity of the drift term, a linear scaling, that is crucial
to simulate  the reverse SDE efficiently and for the proposed Itô density estimators



\textbf{Theorem 1.} \textit{[Itô density estimator] Consider time-dependent density $q_t(x)$ induced by the marginals of the following SDE}
\begin{equation}
    d\bm{x}_t = \bm{f}_t(\bm{x}_t)dt + g_t d\bm{W}_t, \quad \bm{x}_{t=0} \sim q_0(\bm{x}), \quad t \in [0,1], \tag{11}
\end{equation}
\textit{where $d\bm{W}_t$ is the Wiener process. For the reverse-time ($\tau = 1 - t$) SDEs with \textbf{any} vector field $\bm{u}_\tau$ and the same diffusion coefficient $g_t$, i.e.}
\begin{equation}
    d\bm{x}_\tau = \bm{u}_\tau(\bm{x}_\tau)d\tau + g_{1-\tau} d\overline{\bm{W}}_\tau, \quad \tau \in [0,1], \tag{12}
\end{equation}
\textit{the change of the log-density $\log q_\tau(\bm{x}_\tau)$ follows the following SDE}
\begin{equation}
    d \log q_{1-\tau}(\bm{x}_\tau) = \left\langle d\bm{x}_\tau, \nabla \log q_{1-\tau}(\bm{x}_\tau) \right\rangle 
    + \left\langle \nabla, \bm{f}_{1-\tau}(\bm{x}_\tau) \right\rangle 
    + \left\langle \bm{f}_{1-\tau}(\bm{x}_\tau) - \frac{g_{1-\tau}^2}{2} \nabla \log q_{1-\tau}(\bm{x}_\tau), \nabla \log q_{1-\tau}(\bm{x}_\tau) \right\rangle d\tau. \tag{13}
\end{equation}

of the change of log-density includes only the divergence of the forward SDE drift $\left\langle nabla, \bm{f}_{1-\tau}(\bm{x}_\tau) \right\rangle $ <grad, f_1-t(x_t)>.
using the Ornstein-Uhlenbeck SDE, this divergence is simply a constant due to a linear drift scalin